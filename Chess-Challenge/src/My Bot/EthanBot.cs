using System;
using ChessChallenge.API;

public class EthanBot : IChessBot
{
    private const int Depth = 6;
    
    public Move Think(Board board, Timer timer)
    {
        int maxScore = int.MinValue;
        Move? bestMove = null;
        
        Span<Move> moves = stackalloc Move[128];
        board.GetLegalMovesNonAlloc(ref moves);

        foreach (Move move in moves)
        {
            board.MakeMove(move);
            var score = Negate(Search(Depth - 1, board, int.MinValue, int.MaxValue));
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
        
        var max = int.MinValue;
        
        Span<Move> moves = stackalloc Move[128];
        board.GetLegalMovesNonAlloc(ref moves);

        foreach (Move move in moves)
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
            return board.IsWhiteToMove ? int.MinValue : int.MaxValue;
        
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