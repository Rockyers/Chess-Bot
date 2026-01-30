using System;
using ChessChallenge.API;

public class EthanBot : IChessBot
{
    private const int StartingDepth = 6;
    
    // Transposition Table
    private const int TTSize = 1 << 20; // ~1M entries
    private readonly TTEntry[] _tt = new TTEntry[TTSize];
    
    // Killer moves, 2 per depth
    private readonly Move[,] _killerMoves = new Move[32, 2];

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
        var (zKey, _, entry) = GetZobrist(board);

        Span<Move> moves = stackalloc Move[128];
        board.GetLegalMovesNonAlloc(ref moves);
        GetAndOrderMoves(board, moves, entry, zKey);

        var maxScore = int.MinValue;
        var bestMove = moves[0];
        
        foreach (var move in moves)
        {
            board.MakeMove(move);
            var score = Negate(Search(StartingDepth - 1, board, int.MinValue, int.MaxValue));
            board.UndoMove(move);

            if (score <= maxScore) continue;
            
            maxScore = score;
            bestMove = move;
        }

        return bestMove;
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
            {
                UpdateKillerMoves(entry.BestMove, board.PlyCount);
                return entry.Score;
            }
        }
        
        if (depth == 0)
            return RelativeEvaluate(board);
        
        var maxScore = int.MinValue;
        var bestMove = Move.NullMove;
        
        Span<Move> moves = stackalloc Move[128];
        board.GetLegalMovesNonAlloc(ref moves);
        GetAndOrderMoves(board, moves, entry, zKey);

        foreach (var move in moves)
        {
            var ply = board.PlyCount;
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
            {
                UpdateKillerMoves(move, ply);
                break;
            }
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

    private void UpdateKillerMoves(Move move, int ply)
    {
        if (move.IsCapture)
            return;

        if (_killerMoves[ply, 0] == move) return;
        
        _killerMoves[ply, 1] = _killerMoves[ply, 0];
        _killerMoves[ply, 0] = move;
    }

    private void GetAndOrderMoves(Board board, Span<Move> moves, TTEntry ttEntry, ulong zKey)
    {
        var hasTTMove = ttEntry.Key == zKey;

        var writeIndex = 0;
        if (hasTTMove && ttEntry.BestMove != Move.NullMove)
            BubbleMoves(moves, ref writeIndex, move => move == ttEntry.BestMove);

        for (var k = 0; k < 2; k++)
        {
            var k1 = k;
            var killer = _killerMoves[board.PlyCount, k1];
            
            if (killer != Move.NullMove)
                BubbleMoves(moves, ref writeIndex, move => move == killer);
        }

        var startOfCaptures = writeIndex;
        BubbleMoves(moves, ref writeIndex, move => move.IsCapture);
        
        OrderCaptures(moves[startOfCaptures..writeIndex]);
    }

    private static void BubbleMoves(Span<Move> moves, ref int writeIndex, Predicate<Move> predicate)
    {
        for (var i = writeIndex; i < moves.Length; i++)
        {
            if (!predicate.Invoke(moves[i])) continue;
                
            Swap(ref moves[i], ref moves[writeIndex]);
            writeIndex++;
                
            break;
        }
    }
    
    private void OrderCaptures(Span<Move> captures)
    {
        Span<int> scores = stackalloc int[captures.Length];

        for (var i = 0; i < captures.Length; i++)
            scores[i] = CaptureScore(captures[i]);

        for (var i = 0; i < captures.Length; i++)
        {
            var move = captures[i];
            var score = scores[i];

            var j = i - 1;
            while (j >= 0 && scores[j] < score)
            {
                captures[j + 1] = captures[j];
                scores[j + 1] = scores[j];
                j--;
            }

            captures[j + 1] = move;
            scores[j + 1] = score;
        }
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
    private readonly int[] _pieceValues = { 0, 100, 300, 300, 500, 900, 0 };
    private int MaterialWeight(PieceType pieceType, bool isWhite, Board board)
    {
        var bitboard = board.GetPieceBitboard(pieceType, isWhite);
        return CountSetBits(bitboard) * _pieceValues[(int)pieceType];
    }
    
    private int CaptureScore(Move move)
    {
        return _pieceValues[(int)move.CapturePieceType] * 10
               - _pieceValues[(int)move.MovePieceType];
    }
    
    private (ulong zKey, int ttIndex, TTEntry entry) GetZobrist(Board board)
    {
        var zKey = board.ZobristKey;
        var ttIndex = (int)(zKey & (TTSize - 1));
        var entry = _tt[ttIndex];
        return (zKey, ttIndex, entry);
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

    private static int Negate(int i)
    {
        return i switch
        {
            int.MinValue => int.MaxValue,
            int.MaxValue => int.MinValue,
            _ => -i
        };
    }
    
    private static void Swap(ref Move a, ref Move b)
    {
        (a, b) = (b, a);
    }
}