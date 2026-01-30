using System;
using ChessChallenge.API;

public class EthanBot : IChessBot
{
    private const int StartingDepth = 6;
    
    // Transposition Table
    private const int TTSize = 1 << 20; // ~1M entries
    private readonly TTEntry[] _tt = new TTEntry[TTSize];

    private enum TTFlag { Exact, LowerBound, UpperBound }

    private struct TTEntry
    {
        public ulong Key;
        public int Depth;
        public int Score;
        public TTFlag Flag;
        public Move BestMove;
    }
    
    public Move Think(Board board, Timer timer)
    {
        var maxScore = int.MinValue;
        Move? bestMove = null;
        
        var (_, _, entry) = GetZobrist(board);

        Span<Move> moves = stackalloc Move[128];
        GetAndOrderMoves(board, moves, entry);

        foreach (var move in moves)
        {
            board.MakeMove(move);
            var score = Negate(Search(StartingDepth - 1, board, int.MinValue, int.MaxValue));
            board.UndoMove(move);

            if (score <= maxScore) continue;
            
            maxScore = score;
            bestMove = move;
        }
        
        return bestMove ?? Move.NullMove;
    }

    private int Search(int depth, Board board, int alpha, int beta)
    {
        var originalAlpha = alpha;
        
        if (board.IsDraw())
            return 0;

        if (board.IsInCheckmate())
            return int.MinValue + board.PlyCount;

        var (zKey, ttIndex, entry) = GetZobrist(board);

        if (entry.Key == zKey && entry.Depth >= depth)
        {
            switch (entry.Flag)
            {
                case TTFlag.Exact:
                    return entry.Score;
                case TTFlag.LowerBound:
                    alpha = Math.Max(alpha, entry.Score);
                    break;
                case TTFlag.UpperBound:
                    beta = Math.Min(beta, entry.Score);
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            if (alpha >= beta)
                return entry.Score;
        }
        
        if (depth == 0)
            return RelativeEvaluate(board);
        
        var maxScore = int.MinValue;
        var bestMove = Move.NullMove;
        
        Span<Move> moves = stackalloc Move[128];
        GetAndOrderMoves(board, moves, entry);

        foreach (var move in moves)
        {
            board.MakeMove(move);
            var score = Negate(Search(depth - 1, board, Negate(beta), Negate(alpha)));            
            board.UndoMove(move);
            
            if (score > maxScore)
            {
                maxScore = score;
                bestMove = move;
            }
            
            if (score > alpha)
                alpha = score;

            if (alpha >= beta)
                break;
        }

        var flag = (maxScore <= originalAlpha, maxScore >= beta) switch
        {
            (true, _) => TTFlag.UpperBound,
            (_, true) => TTFlag.LowerBound,
            _ => TTFlag.Exact
        };
        
        // if entry is EMPTY or depth is higher (more accurate eval)
        if (entry.Key != zKey || depth >= entry.Depth)
        {
            _tt[ttIndex] = new TTEntry
            {
                Key = zKey,
                Depth = depth,
                Score = maxScore,
                Flag = flag,
                BestMove = bestMove
            };
        }
        
        return maxScore;
    }

    private static void GetAndOrderMoves(Board board, Span<Move> moves, TTEntry ttEntry)
    {
        board.GetLegalMovesNonAlloc(ref moves);
        
        Span<int> scores = stackalloc int[moves.Length];
        for (var i = 0; i < moves.Length; i++)
        {
            var move = moves[i];
            var score = EvaluateMove(ttEntry, move);
            scores[i] = score;
        }
        
        // INSERTION SORT BY SCORE
        for (var i = 1; i < moves.Length; i++)
        {
            var move = moves[i];
            var score = scores[i];

            var j = i - 1;
            while (j >= 0 && scores[j] < score)
            {
                moves[j + 1] = moves[j];
                scores[j + 1] = scores[j];
                j--;
            }

            moves[j + 1] = move;
            scores[j + 1] = score;
        }
    }

    private static int EvaluateMove(TTEntry ttEntry, Move move)
    {
        var score = 0;

        if (move == ttEntry.BestMove)
            score = 1000000;

        if (move.IsCapture)
            score += 10000 + (int) move.CapturePieceType * 100 - (int) move.MovePieceType;
        
        return score;
    }

    private int RelativeEvaluate(Board board)
    {
        var eval = EvaluateBoard(board);

        return board.IsWhiteToMove ? eval : Negate(eval);
    }

    private int EvaluateBoard(Board board)
    {
        if (board.IsInStalemate())
            return 0;
        
        if (board.IsInCheckmate())
            return board.IsWhiteToMove 
                ? int.MinValue + board.PlyCount 
                : int.MaxValue - board.PlyCount;
     
        var eval = 0;
        
        for (var i = 1; i < 6; i++)
        {
            var pieceType = (PieceType) i;
            eval += MaterialWeight(pieceType, true, board) - MaterialWeight(pieceType, false, board);
        }

        return eval;
    }
    
    // Piece values: null, pawn, knight, bishop, rook, queen
    private readonly int[] _pieceValues = { 0, 100, 300, 300, 500, 900 };
    private int MaterialWeight(PieceType pieceType, bool isWhite, Board board)
    {
        var bitboard = board.GetPieceBitboard(pieceType, isWhite);
        return CountSetBits(bitboard) * _pieceValues[(int)pieceType];
    }
    
    private static int CountSetBits(ulong n)
    {
        var count = 0;
        while (n != 0)
        {
            count++;
            n &= n - 1;
        }
        return count;
    }
    
    private (ulong zKey, int ttIndex, TTEntry entry) GetZobrist(Board board)
    {
        var zKey = board.ZobristKey;
        var ttIndex = (int)(zKey & (TTSize - 1));
        var entry = _tt[ttIndex];
        return (zKey, ttIndex, entry);
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