using System;
using System.Numerics;
using ChessChallenge.API;

namespace Chess_Challenge.My_Bot.Ethan_Bot;

public class EthanBot : IChessBot
{
    public string Name() => "Ethan Bot";
    
    private const int MaximumDepth = 30;
    private const int MaxQuiescenceDepth = 15;
    private const int AspirationWindow = 50;

    private const int MinValue = -100_000_000;
    private const int MaxValue =  100_000_000;

    private Move _previousBestMove = Move.NullMove;
    
    // Transposition Table
    private const int TTSize = 1 << 22; // ~4M entries
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
    
    // Killer moves, 2 per depth
    private readonly Move[,] _killerMoves = new Move[256, 2];
    
    // History Heuristic
    private readonly int[,,] _history = new int[2, 64, 64];
    
    public Move Think(Board board, Timer timer)
    {
        var timeLimit = timer.MillisecondsRemaining / 40;

        // Decay History
        if (board.PlyCount % 6 == 0 && board.PlyCount > 1) 
        {
            for (var s = 0; s < 2; s++)
                for (var f = 0; f < 64; f++)
                    for (var t = 0; t < 64; t++)
                        _history[s, f, t] /= 2;
        }
        
        var (zKey, _, entry) = GetZobrist(board);

        Span<Move> moves = stackalloc Move[128];
        board.GetLegalMovesNonAlloc(ref moves);
        OrderMoves(board, moves, entry, zKey);
        
        var bestMove = moves[0];

        var d = 0;
        var lastScore = 0;
        
        for (var depth = 1; depth < MaximumDepth; depth++)
        {
            if (timer.MillisecondsElapsedThisTurn >= timeLimit)
                break;
            
            d = depth;
            
            var alpha = Math.Max(MinValue + 1, lastScore - AspirationWindow);
            var beta = Math.Min(MaxValue - 1, lastScore + AspirationWindow);
            
            var maxScore = MinValue;

            if (_previousBestMove != Move.NullMove)
            {
                for (var i = 0; i < moves.Length; i++)
                {
                    if (moves[i] != _previousBestMove) continue;
                
                    Swap(ref moves[i], ref moves[0]);
                    break;
                }
            }
            
            OrderMoves(board, moves, entry, zKey, true);
            
            foreach (var move in moves)
            {
                if (timer.MillisecondsElapsedThisTurn >= timeLimit)
                    break;
            
                board.MakeMove(move);
                var score = -Search(depth, board, alpha, beta);
                board.UndoMove(move);

                if (score <= maxScore) continue;
            
                maxScore = score;
                bestMove = move;
            }

            if (maxScore <= alpha || maxScore >= beta)
            {
                maxScore = MinValue;
                foreach (var move in moves)
                {
                    if (timer.MillisecondsElapsedThisTurn >= timeLimit)
                        break;

                    board.MakeMove(move);
                    var score = -Search(depth, board, MinValue, MaxValue);
                    board.UndoMove(move);

                    if (score <= maxScore) continue;

                    maxScore = score;
                    bestMove = move;
                }
            }

            lastScore = maxScore;
            _previousBestMove = bestMove;
        }

        Console.WriteLine("Reached a depth of " + d);
        
        return bestMove;
    }

    // **** SEARCH ****
    private int Search(int depth, Board board, int alpha, int beta)
    {
        var originalAlpha = alpha;
        
        if (board.IsDraw())
            return 0;

        if (board.IsInCheckmate())
            return MinValue + board.PlyCount;

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
            }

            if (alpha >= beta)
                return alpha;
        }
        
        // NULL MOVE PRUNING
        if (!board.IsInCheck() && depth > 3 && HasNonPawnMaterial(board))
        {
            board.TrySkipTurn();
            var score = -Search(depth - 3, board, -beta, -beta + 1);
            board.UndoSkipTurn();

            if (score >= beta)
                // HUGE skip whole branch
                return beta;
        }
        
        if (depth == 0)
            return Quiescence(board, alpha, beta);
        
        var maxScore = MinValue;
        var bestMove = Move.NullMove;
        
        Span<Move> moves = stackalloc Move[128];
        board.GetLegalMovesNonAlloc(ref moves);
        OrderMoves(board, moves, entry, zKey);

        var moveIndex = 0;
        foreach (var move in moves)
        {
            var ply = board.PlyCount;
            board.MakeMove(move);

            var score = 0;
            var nextDepth = depth - 1;

            var reduce = moveIndex >= 4 && depth >= 3 && !board.IsInCheck() && !move.IsCapture;

            if (reduce)
            {
                score = -Search(nextDepth - 1, board, -alpha - 1, -alpha);

                if (score > alpha)
                    reduce = false;
            }

            if (!reduce)
            {
                score = -Search(nextDepth, board, -beta, -alpha);
            }
            
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
                UpdateQuietCutoff(move, ply, board.IsWhiteToMove, depth);
                break;
            }

            moveIndex++;
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
    
    private void UpdateQuietCutoff(Move move, int ply, bool isWhiteToPlay, int depth)
    {
        if (move.IsCapture)
            return;

        if (_killerMoves[ply, 0] != move)
        {
            _killerMoves[ply, 1] = _killerMoves[ply, 0];
            _killerMoves[ply, 0] = move;
        }

        var side = isWhiteToPlay ? 0 : 1;
        _history[side, move.StartSquare.Index, move.TargetSquare.Index] += depth * depth;
    }
    
    private void OrderMoves(Board board, Span<Move> moves, TTEntry ttEntry, ulong zKey, bool root = false)
    {
        Span<int> scores = stackalloc int[moves.Length];

        var ply = board.PlyCount;

        var ttMove = (ttEntry.Key == zKey) ? ttEntry.BestMove : Move.NullMove;
        var killer0 = _killerMoves[ply, 0];
        var killer1 = _killerMoves[ply, 1];

        var side = board.IsWhiteToMove ? 0 : 1;

        for (var i = 0; i < moves.Length; i++)
        {
            var move = moves[i];
            var score = 0;

            if (root && move == _previousBestMove && _previousBestMove != Move.NullMove)
                score += 1500000;

            if (move == ttMove)
                score += 1000000;
            
            else if (move == killer0)
                score += 900000;
            else if (move == killer1)
                score += 800000;

            if (move.IsCapture)
                score += CaptureScore(move);
            else
                score += _history[side, move.StartSquare.Index, move.TargetSquare.Index];
            
            scores[i] = score;
        }
        
        // Insertion Sort
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

    private int Quiescence(Board board, int alpha, int beta, int qDepth = 1)
    {
        var eval = RelativeEvaluate(board);

        if (qDepth > MaxQuiescenceDepth)
            return eval;
        
        if (eval >= beta)
            return beta;
        if (eval > alpha)
            alpha = eval;
        
        Span<Move> moves = stackalloc Move[128];
        board.GetLegalMovesNonAlloc(ref moves, true);
        OrderCaptures(moves, 0, moves.Length);

        foreach (var move in moves)
        {
            board.MakeMove(move);
            var score = -Quiescence(board, -beta, -alpha, qDepth + 1);
            board.UndoMove(move);

            if (score >= beta)
                return beta;
            if (score > alpha)
                alpha = score;
        }

        return alpha;
    }
    
    private void OrderCaptures(Span<Move> captures, int start, int end)
    {
        Span<int> scores = stackalloc int[captures.Length];

        for (var i = start; i < end; i++)
            scores[i] = CaptureScore(captures[i]);

        for (var i = start; i < end; i++)
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
    
    private (ulong zKey, int ttIndex, TTEntry entry) GetZobrist(Board board)
    {
        var zKey = board.ZobristKey;
        var ttIndex = (int)(zKey & (TTSize - 1));
        var entry = _tt[ttIndex];
        return (zKey, ttIndex, entry);
    }
    
    private static bool HasNonPawnMaterial(Board board)
    {
        var white = board.IsWhiteToMove;

        // Knights, bishops, rooks, queens
        return
            board.GetPieceBitboard(PieceType.Knight, white) != 0 ||
            board.GetPieceBitboard(PieceType.Bishop, white) != 0 ||
            board.GetPieceBitboard(PieceType.Rook,   white) != 0 ||
            board.GetPieceBitboard(PieceType.Queen,  white) != 0;
    }

    // **** EVALUATE ****
    // Piece values: null, pawn, knight, bishop, rook, queen
    private readonly int[] _pieceValues = [0, 100, 320, 330, 500, 900, 0];
    private readonly int[] _pieceValuesEndgame = [0, 120, 310, 330, 500, 900, 0];
    private readonly int[] _phaseValue = [0,   0,   1,   1,   2,   4, 0];
    
    private int RelativeEvaluate(Board board)
    {
        var eval = EvaluateBoard(board);

        return board.IsWhiteToMove ? eval : -eval;
    }

    private int EvaluateBoard(Board board)
    {
        if (board.IsInStalemate())
            return 0;
        
        if (board.IsInCheckmate())
            return board.IsWhiteToMove 
                ? MinValue + board.PlyCount 
                : MaxValue - board.PlyCount;
     
        var middleGameEval = 0;
        var endGameEval = 0;
        var phase = ComputePhase(board);

        EvaluateMaterial(board, ref middleGameEval, ref endGameEval);
        EvaluatePieceSquares(board, ref middleGameEval, ref endGameEval);

        return (middleGameEval * phase + endGameEval * (24 - phase)) / 24;
    }

    private void EvaluatePieceSquares(Board board, ref int middleGameEval, ref int endGameEval)
    {
        throw new NotImplementedException();
    }

    private void EvaluateMaterial(Board board, ref int middleGameEval, ref int endGameEval)
    {
        for (var p = 0; p < 6; p++)
        {
            middleGameEval += MaterialWeight((PieceType)p, true, board, _pieceValues) -
                              MaterialWeight((PieceType)p, false, board, _pieceValues);
            
            endGameEval += MaterialWeight((PieceType)p, true, board, _pieceValuesEndgame) -
                           MaterialWeight((PieceType)p, false, board, _pieceValuesEndgame);
        }
    }
    
    private static int MaterialWeight(PieceType pieceType, bool isWhite, Board board, int[] pieceValues)
    {
        var bitboard = board.GetPieceBitboard(pieceType, isWhite);
        return CountSetBits(bitboard) * pieceValues[(int)pieceType];
    }

    // Calculates the phase of the game from 24 -> start/middle game, to 0 -> endgame
    private int ComputePhase(Board board)
    {
        var phase = 0;

        for (var piece = 0; piece < 6; piece++)
        {
            phase += _phaseValue[piece] * (CountSetBits(board.GetPieceBitboard((PieceType)piece, true)) + 
                                           CountSetBits(board.GetPieceBitboard((PieceType)piece, false)));
        }

        return Math.Min(phase, 24);
    }
    
    // How "worth it" it is to take a piece (pawn taking a queen > queen taking a pawn)
    private int CaptureScore(Move move)
    {
        return _pieceValues[(int)move.CapturePieceType] * 10
               - _pieceValues[(int)move.MovePieceType];
    }

    private static int CountSetBits(ulong n) => BitOperations.PopCount(n);
    private static void Swap(ref Move a, ref Move b) => (a, b) = (b, a);
}