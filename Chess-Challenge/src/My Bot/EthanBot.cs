using System;
using ChessChallenge.API;

public class EthanBot : IChessBot
{
    private enum TTFlag : byte
    {
        Exact,
        LowerBound,
        UpperBound
    }

    private struct TTEntry
    {
        public ulong Key;
        public int Depth;
        public int Score;
        public TTFlag Flag;
        public Move BestMove;
    }

    private const int TTSize = 1 << 20; // ~1M entries
    private TTEntry[] _tt = new TTEntry[TTSize];
    
    private ref TTEntry Probe(ulong key)
    {
        return ref _tt[key & (TTSize - 1)];
    }
    
    private const int startingDepth = 6;
    
    public Move Think(Board board, Timer timer)
    {
        var maxScore = int.MinValue;
        Move? bestMove = null;
        
        Span<Move> moves = stackalloc Move[128];
        board.GetLegalMovesNonAlloc(ref moves);

        foreach (var move in moves)
        {
            board.MakeMove(move);
            var score = Negate(Search(startingDepth - 1, board, int.MinValue, int.MaxValue));
            board.UndoMove(move);

            if (score > maxScore)
            {
                maxScore = score;
                bestMove = move;
            }
        }
        
        return bestMove ?? Move.NullMove;
    }

    private int Search(int depth, Board board, int alpha, int beta)
    {
        if (board.IsDraw())
            return 0;

        if (board.IsInCheckmate())
            return int.MinValue;
        
        if (depth == 0)
            return RelativeEvaluate(board);
        
        var key = board.ZobristKey;
        ref var entry = ref Probe(key);

        if (entry.Key == key && entry.Depth >= depth)
        {
            switch (entry.Flag)
            {
                case TTFlag.Exact:
                case TTFlag.LowerBound when entry.Score >= beta:
                case TTFlag.UpperBound when entry.Score <= alpha:
                    return entry.Score;
                case TTFlag.LowerBound:
                    alpha = Math.Max(alpha, entry.Score);
                    break;
                case TTFlag.UpperBound:
                    beta = Math.Min(beta, entry.Score);
                    break;
            }
        }
        
        var max = int.MinValue;
        var originalAlpha = alpha;
        
        Span<Move> moves = stackalloc Move[128];
        board.GetLegalMovesNonAlloc(ref moves);

        foreach (var move in moves)
        {
            board.MakeMove(move);
            var score = Negate(Search(depth - 1, board, Negate(beta), Negate(alpha)));            
            board.UndoMove(move);
            
            if (score > max)
            {
                max = score;
                if (score > alpha)
                    alpha = score;
            }
            
            if (score >= beta)
                return score;
        }

        return max;
    }

    private int RelativeEvaluate(Board board)
    {
        var eval = Evaluate(board);

        if (board.IsWhiteToMove)
            return eval;

        return Negate(eval);
    }

    private int Evaluate(Board board)
    {
        var eval = 0;

        if (board.IsInStalemate())
            return 0;
        
        if (board.IsInCheckmate())
            return board.IsWhiteToMove ? int.MinValue + board.PlyCount : int.MaxValue - board.PlyCount;
        
        for (var i = 1; i < 6; i++)
        {
            var pieceType = (PieceType) i;
            eval += MaterialWeight(pieceType, true, board) - MaterialWeight(pieceType, false, board);
        }

        return eval;
    }
    
    // Piece values: null, pawn, knight, bishop, rook, queen
    private readonly int[] _pieceValues = { 0, 100, 300, 350, 500, 900 };
    private int MaterialWeight(PieceType pieceType, bool isWhite, Board board)
    {
        ulong bitboard = board.GetPieceBitboard(pieceType, isWhite);
        return CountSetBits(bitboard) * _pieceValues[(int)pieceType];
    }
    
    private static int CountSetBits(ulong n)
    {
        ulong num = n;
        
        int count = 0;
        while (num != 0)
        {
            count++;
            num &= num - 1;
        }
        return count;
    }

    private static int Negate(int i)
    {
        return i switch
        {
            int.MinValue => int.MaxValue,
            int.MaxValue => int.MinValue,
            _ => -i
        };
    }
}