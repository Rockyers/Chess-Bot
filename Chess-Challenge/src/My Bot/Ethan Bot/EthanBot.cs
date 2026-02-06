using System;
using System.Numerics;
using ChessChallenge.API;
using ChessChallenge.Application;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Chess_Challenge.My_Bot.Ethan_Bot;

public class EthanBot : IChessBot
{
    public string Name() => "Ethan Bot";
    
    private const int MaximumDepth = 30;
    private const int MaxQuiescenceDepth = 15;
    private const int AspirationWindow = 80;

    private const int MinValue = -1_000_000_000;
    private const int MaxValue =  1_000_000_000;

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
    private readonly Move[,] _killerMoves = new Move[512, 2];
    
    // History Heuristic
    private readonly int[,,] _history = new int[2, 64, 64];
    
    public Move Think(Board board, Timer timer)
    {
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
        
        var timeLimit = ComputeTimeLimit(board, timer, moves);
        
        var bestMove = moves[0];

        var depthLog = 0;
        var lastScore = 0;
        
        for (var depth = 1; depth < MaximumDepth; depth++)
        {
            if (timer.MillisecondsElapsedThisTurn >= timeLimit)
                break;
            
            depthLog = depth;

            var window = depth < 4 ? 200 : AspirationWindow;
            OrderMoves(board, moves, entry, zKey, true);

            while (true)
            {
                if (timer.MillisecondsElapsedThisTurn >= timeLimit)
                    break;
                
                var alpha = Math.Max(MinValue + 1, lastScore - window);
                var beta = Math.Min(MaxValue - 1, lastScore + window);
                
                var maxScore = MinValue;
                var windowBest = bestMove;
                
                foreach (var move in moves)
                {
                    if (timer.MillisecondsElapsedThisTurn >= timeLimit)
                        break;
            
                    board.MakeMove(move);
                    var score = -Search(depth - 1, board, -beta, -alpha);
                    board.UndoMove(move);

                    if (score > maxScore)
                    {
                        maxScore = score;
                        windowBest = move;
                    }

                    if (score > alpha)
                    {
                        alpha = score;
                    }
                }

                if (maxScore > lastScore - window && maxScore < lastScore + window)
                {
                    lastScore = maxScore;
                    bestMove = windowBest;
                    _previousBestMove = bestMove;
                    break;
                }

                window += window / 2;

                // Just in case
                if (window > 10000)
                {
                    lastScore = maxScore;
                    bestMove = windowBest;
                    _previousBestMove = bestMove;
                    break;
                }
            }
        }

        ConsoleHelper.Log("Reached a depth of " + depthLog);
        
        return bestMove;
    }

    // Via ChatGPT
    private static int ComputeTimeLimit(Board board, Timer timer, Span<Move> moves)
    {
        var movesLeft = board.PlyCount < 40 ? 40 - board.PlyCount / 2 : 20; // Estimate
        var baseTime = timer.MillisecondsRemaining / (double) movesLeft;
        
        // Scale by complexity: more captures -> spend more time
        var moveCount = moves.Length;
        
        var complexityFactor = Math.Min(2.0, 1.0 + (moveCount / 20.0)); // 1–2x

        // Scale by phase: endgame moves require less time generally
        var phase = ComputePhase(board) / 24.0; // 1=middlegame, 0=endgame
        var phaseFactor = 0.8 + 0.4 * phase; // 0.8–1.2x

        var timeLimit = (int)(baseTime * complexityFactor * phaseFactor);
        return Math.Max(5, timeLimit); // at least 5ms per depth
    }

    // **** SEARCH ****
    private int Search(int depth, Board board, int alpha, int beta)
    {
        var originalAlpha = alpha;
        var isWhite = board.IsWhiteToMove;
        
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
        if (!board.IsInCheck() && 
            depth >= 4 && 
            ComputePhase(board) > 14 && 
            (board.GetPieceBitboard(PieceType.Queen, isWhite) != 0 || board.GetPieceBitboard(PieceType.Rook, isWhite) != 0) &&
            CountSetBits(board.GetPieceBitboard(PieceType.Pawn, isWhite)) > 3)
        {
            board.TrySkipTurn();
            var r = depth > 6 ? 3 : 2;
            var score = -Search(depth - r, board, -beta, -beta + 1);
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

            var reduce = moveIndex >= 4 && depth >= 4 && !board.IsInCheck() && !move.IsCapture;

            var reduction = 0;
            if (reduce)
            {
                // Dynamic formula based on online sources, smth like stockfish
                reduction = 1 + (depth * moveIndex) / 16;
                reduction = Math.Min(reduction, depth - 1); // Prevent reducing to 0

                var historyScore = _history[isWhite ? 0 : 1, move.StartSquare.Index, move.TargetSquare.Index];
                if (move == _killerMoves[ply, 0] || move == _killerMoves[ply, 1] || historyScore > 200 + 5 * depth)
                    reduction = 0; // Dont reduce good history moves or killer moves
            }
            
            if (reduction > 0)
            {
                score = -Search(nextDepth - reduction, board, -alpha - 1, -alpha);

                if (score > alpha)
                    score = -Search(nextDepth, board, -beta, -alpha);
            }
            else
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
                UpdateQuietCutoff(move, ply, isWhite, depth);
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

    private static int Quiescence(Board board, int alpha, int beta, int qDepth = 1)
    {
        var eval = EvaluateRelative(board);

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
            if (CaptureScore(move) < 0) continue;
            
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
    
    private static void OrderCaptures(Span<Move> captures, int start, int end)
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

    // **** EVALUATION ****
    // Piece values: null, pawn, knight, bishop, rook, queen
    private static readonly int[] PieceValues = [0, 100, 320, 330, 500, 900, 0];
    private static readonly int[] PieceValuesEndgame = [0, 120, 310, 330, 500, 900, 0];
    private static readonly int[] PhaseValue = [0, 0, 1, 1, 2, 4, 0];

    private static readonly int[] MobilityMiddlegame = [0, 0, 4, 5, 2, 1, 0];
    private static readonly int[] MobilityEndgame = [0, 0, 2, 3, 2, 1, 0];

    private static readonly int[] PassedPawnsMiddlegame = [0, 0, 5, 10, 20, 35, 60, 0];
    private static readonly int[] PassedPawnsEndgame = [0, 0, 20, 40, 70, 120, 200, 0];

    private static readonly ulong[] FileMasks =
    [
        0x0101010101010101UL,
        0x0202020202020202UL,
        0x0404040404040404UL,
        0x0808080808080808UL,
        0x1010101010101010UL,
        0x2020202020202020UL,
        0x4040404040404040UL,
        0x8080808080808080UL
    ];

    private static readonly int[] KingSafetyPenaltyMiddlegame = [0, 10, 30, 50];
    private static readonly int[] KingSafetyPenaltyEndgame = [0, 5, 15, 25];

    public static float DisplayEval(Board board)
    {
        return (board.IsWhiteToMove ? 1 : -1) * Quiescence(board, MinValue, MaxValue) / 100.0f;
    }
    
    private static int EvaluateRelative(Board board)
    {
        var eval = EvaluateBoard(board);

        return board.IsWhiteToMove ? eval : -eval;
    }

    private static int EvaluateBoard(Board board)
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
        if (phase > 7) EvaluateKingSafety(board, ref middleGameEval, ref endGameEval);
        EvaluateDoubleBishop(board, ref middleGameEval, ref endGameEval);
        if (phase > 5) EvaluateMobility(board, ref middleGameEval, ref endGameEval);
        
        // Expensive so reserve for opening/later in the game after pieces are gone
        if (board.PlyCount < 4 || phase < 22)
            EvaluatePawnStructure(board, ref middleGameEval, ref endGameEval);
        
        EvaluatePassedPawns(board, ref middleGameEval, ref endGameEval);
        EvaluateTempo(board, ref middleGameEval, ref endGameEval);

        return (middleGameEval * phase + endGameEval * (24 - phase)) / 24;
    }

    private static void EvaluateMaterial(Board board, ref int middleGameEval, ref int endGameEval)
    {
        for (var piece = 1; piece < 6; piece++)
        {
            middleGameEval += MaterialWeight((PieceType)piece, true, board, PieceValues) -
                              MaterialWeight((PieceType)piece, false, board, PieceValues);
            
            endGameEval += MaterialWeight((PieceType)piece, true, board, PieceValuesEndgame) -
                           MaterialWeight((PieceType)piece, false, board, PieceValuesEndgame);
        }
    }
    
    private static int MaterialWeight(PieceType pieceType, bool isWhite, Board board, int[] pieceValues)
    {
        var bitboard = board.GetPieceBitboard(pieceType, isWhite);
        return CountSetBits(bitboard) * pieceValues[(int)pieceType];
    }
    
    private static void EvaluatePieceSquares(Board board, ref int middleGameEval, ref int endGameEval)
    {
        // White
        for (var piece = 1; piece < 7; piece++)
        {
            var pieceType = (PieceType) piece;
            var bitboard = board.GetPieceBitboard(pieceType, true);
            while (bitboard != 0)
            {
                var square = BitOperations.TrailingZeroCount(bitboard);
                bitboard &= bitboard - 1; // Clear the rightmost set bit

                middleGameEval += PieceSquareTables.GetSquare(piece, square, false);
                endGameEval += PieceSquareTables.GetSquare(piece, square, true);
            }
        }
        
        //Black
        for (var piece = 1; piece < 7; piece++)
        {
            var pieceType = (PieceType) piece;
            var bitboard = board.GetPieceBitboard(pieceType, false);
            while (bitboard != 0)
            {
                var square = BitOperations.TrailingZeroCount(bitboard) ^ 0b111000; //XOR with 111000 (56) to flip in groups of eight (flip the board) 
                bitboard &= bitboard - 1; // Clear the rightmost set bit

                middleGameEval -= PieceSquareTables.GetSquare(piece, square, false);
                endGameEval -= PieceSquareTables.GetSquare(piece, square, true);
            }
        }
    }

    private static void EvaluateKingSafety(Board board, ref int middleGameEval, ref int endGameEval)
    {
        EvaluateKingSafetySide(board, true, ref middleGameEval, ref endGameEval);
        EvaluateKingSafetySide(board, false, ref middleGameEval, ref endGameEval);
    }

    private static void EvaluateKingSafetySide(Board board, bool isWhite, ref int middleGameEval, ref int endGameEval)
    {
        var kingBoard = board.GetPieceBitboard(PieceType.King, isWhite);
        if (kingBoard == 0) return; // Should never happen

        var mgPenalty = 0;
        var egPenalty = 0;
        
        var square =  BitOperations.TrailingZeroCount(kingBoard);

        var kingBox = KingBox(square);
        
        var enemyPieces = board.GetPieceBitboard(PieceType.Queen, !isWhite) | 
                              board.GetPieceBitboard(PieceType.Rook, !isWhite) | 
                              board.GetPieceBitboard(PieceType.Bishop, !isWhite) | 
                              board.GetPieceBitboard(PieceType.Knight, !isWhite);
        
        var attackers = CountSetBits(enemyPieces & kingBox);
        mgPenalty +=  attackers * 15;
        egPenalty +=  attackers * 5;

        var file = square & 7;
        var rank = square >> 3;
        
        var pawnMask = FileMasks[file];
        if (file > 0) pawnMask |= FileMasks[file - 1];
        if (file < 7) pawnMask |= FileMasks[file + 1];
        
        pawnMask = isWhite 
            ? pawnMask << ((rank + 1) * 8) 
            :  pawnMask >> ((7 - rank) * 8);
        
        var pawns = CountSetBits(board.GetPieceBitboard(PieceType.Pawn, isWhite) & pawnMask);
        var missingPawns = 3 - pawns;

        mgPenalty += KingSafetyPenaltyMiddlegame[Math.Clamp(missingPawns, 0, 3)];
        egPenalty += KingSafetyPenaltyEndgame[Math.Clamp(missingPawns, 0, 3)];

        middleGameEval += mgPenalty * (isWhite ? -1 : 1);
        endGameEval += egPenalty * (isWhite ? -1 : 1);
    }
    
    // Via ChatGPT
    private static ulong KingBox(int square)
    {
        var rank = square >> 3;
        var file = square & 7;

        var bb = 0UL;
        for (var r = Math.Max(0, rank - 1); r <= Math.Min(7, rank + 1); r++)
        for (var f = Math.Max(0, file - 1); f <= Math.Min(7, file + 1); f++)
            bb |= 1UL << (r * 8 + f);

        return bb;
    }

    private static void EvaluateDoubleBishop(Board board, ref int middleGameEval, ref int endGameEval)
    {
        if (CountSetBits(board.GetPieceBitboard(PieceType.Bishop, true)) >= 2)
        {
            middleGameEval += 30;
            endGameEval += 50;
        }

        if (CountSetBits(board.GetPieceBitboard(PieceType.Bishop, false)) >= 2)
        {
            middleGameEval -= 30;
            endGameEval -= 50;
        }
    }
    
    private static void EvaluateMobility(Board board, ref int middleGameEval, ref int endGameEval)
    {
        Span<Move> moves = stackalloc Move[128];
        board.GetLegalMovesNonAlloc(ref moves);
        EvaluateMobility(moves, board.IsWhiteToMove, ref middleGameEval, ref endGameEval);
    }

    private static void EvaluateMobility(Span<Move> moves, bool isWhite, ref int middleGameEval, ref int endGameEval)
    {
        Span<int> count = stackalloc int[7];
        
        foreach (var move in moves)
        {
            var piece = (int) move.MovePieceType;

            // Do NOT evaluate pawn or king mobility
            if (piece is < 2 or > 5) continue;

            count[piece]++;
        }

        for (var piece = 2; piece < 6; piece++)
        { 
            middleGameEval += count[piece] * MobilityMiddlegame[piece] * (isWhite ? 1 : -1);
            endGameEval += count[piece] * MobilityEndgame[piece] * (isWhite ? 1 : -1);
        }
    }

    private static void EvaluatePawnStructure(Board board, ref int middleGameEval, ref int endGameEval)
    {
        var whitePawns = board.GetPieceBitboard(PieceType.Pawn, true);
        var blackPawns = board.GetPieceBitboard(PieceType.Pawn, false);
        
        var whiteScore = EvaluatePawnStructure(whitePawns, blackPawns, true);
        var blackScore = EvaluatePawnStructure(blackPawns, whitePawns, false);

        middleGameEval += whiteScore;
        middleGameEval -= blackScore;
        
        endGameEval += whiteScore / 2;
        endGameEval -= blackScore / 2;
    }

    private static int _doubledPawnPenalty = 15;
    private static int _isolatedPawnPenalty = 20;
    private static int _backwardsPawnPenalty = 10;
    private static int _pawnChainReward = 5;

    private static int EvaluatePawnStructure(ulong pawns, ulong opponentPawns, bool isWhite)
    {
        var score = 0;

        for (var file = 0; file < 8; file++)
        {
            var pawnsOnFile = pawns & FileMasks[file];
            var count = CountSetBits(pawnsOnFile);

            // Pawns doubled up
            if (count > 1)
                score -= _doubledPawnPenalty * (count - 1);

            // Isolated pawns
            var neighborMask = NeighborFileMask(file);
            if ((pawns & neighborMask) == 0 && count > 0)
                score -= _isolatedPawnPenalty * count;

            if (pawnsOnFile == 0) continue;
            
            // Backward pawns
            var frontMask = FileMasks[file];
            frontMask = isWhite
                ? frontMask << 8
                : frontMask >> 8;

            frontMask |= isWhite ? frontMask << 8 : frontMask >> 8;
            frontMask |= isWhite ? frontMask << 16 : frontMask >> 16;
            frontMask |= isWhite ? frontMask << 32 : frontMask >> 32;

            var enemyAttacks = frontMask & opponentPawns & NeighborFileMask(file);
            var friendlySupport = pawns & NeighborFileMask(file);

            if (enemyAttacks != 0 && friendlySupport == 0)
                score -= _backwardsPawnPenalty * count;
            
            // Pawn chains
            var diagonalSupport = isWhite
                ? (pawnsOnFile << 7 & pawns) | (pawnsOnFile << 9 & pawns) // pawns behind either diagonal
                : (pawnsOnFile >> 7 & pawns) | (pawnsOnFile >> 9 & pawns);
            score += _pawnChainReward * CountSetBits(diagonalSupport);
        }

        return score;
    }

    private static ulong FrontMask(ulong pawns, bool isWhite)
    {
        var mask = pawns;
        
        if (isWhite)
        {
            mask |= mask << 8;
            mask |= mask << 16;
            mask |= mask << 32;
        }
        else
        {
            mask |= mask >> 8;
            mask |= mask >> 16;
            mask |= mask >> 32;
        }
        
        return mask;
    }

    private static void EvaluatePassedPawns(Board board, ref int middleGameEval, ref int endGameEval)
    {
        EvaluatePassedPawns(board, true, ref middleGameEval, ref endGameEval);
        EvaluatePassedPawns(board, false, ref middleGameEval, ref endGameEval);
    }

    private static void EvaluatePassedPawns(Board board, bool isWhite, ref int middleGameEval, ref int endGameEval)
    {
        var bitboard = board.GetPieceBitboard(PieceType.Pawn, isWhite);
        var enemyBitboard = board.GetPieceBitboard(PieceType.Pawn, !isWhite);

        while (bitboard != 0)
        {
            var square = BitOperations.TrailingZeroCount(bitboard);
            bitboard &= bitboard - 1;

            var file = square & 7;
            var rank = square >> 3;

            var pawnRank = isWhite ? rank + 1 : 8 - (rank + 1);
            
            if (!IsPassedPawn(enemyBitboard, rank, file, isWhite))
                continue;
            
            var enemyKing = board.GetPieceBitboard(PieceType.King, !isWhite);
            var enemyKingSq = BitOperations.TrailingZeroCount(enemyKing);

            var dist = Math.Abs((enemyKingSq & 7) - file) +
                       Math.Abs((enemyKingSq >> 3) - rank);

            endGameEval += (7 - dist) * (isWhite ? 2 : -2);
            
            middleGameEval += PassedPawnsMiddlegame[pawnRank] * (isWhite ? 1 : -1);
            endGameEval += PassedPawnsEndgame[pawnRank] * (isWhite ? 1 : -1);
        }
    }

    private static bool IsPassedPawn(ulong enemyBitboard, int rank, int file, bool isWhite)
    {
        var mask = FileMasks[file];
        if (file > 0) mask |= FileMasks[file - 1];
        if (file < 7) mask |= FileMasks[file + 1];
        
        mask = isWhite 
            ? mask << ((rank + 1) * 8) 
            :  mask >> ((7 - rank) * 8);

        return (enemyBitboard & mask) == 0;
    }

    private static ulong NeighborFileMask(int file)
    {
        ulong mask = 0;
        if (file > 0) mask |= FileMasks[file - 1];
        if (file < 7) mask |= FileMasks[file + 1];
        return mask;
    }

    private static void EvaluateTempo(Board board, ref int middleGameEval, ref int endGameEval)
    {
        // Minor bonus to the current turn as a "tempo bonus"
        middleGameEval += board.IsWhiteToMove ? 12 : -12;
        endGameEval += board.IsWhiteToMove ? 1 : -1;
    }

    // Calculates the phase of the game from 24 -> start/middle game, to 0 -> endgame
    private static int ComputePhase(Board board)
    {
        var phase = 0;

        for (var piece = 1; piece < 6; piece++)
        {
            phase += PhaseValue[piece] * (CountSetBits(board.GetPieceBitboard((PieceType)piece, true)) + 
                                           CountSetBits(board.GetPieceBitboard((PieceType)piece, false)));
        }

        return Math.Min(phase, 24);
    }
    
    // How "worth it" it is to take a piece (pawn taking a queen > queen taking a pawn)
    private static int CaptureScore(Move move)
    {
        return PieceValues[(int)move.CapturePieceType] * 10
               - PieceValues[(int)move.MovePieceType];
    }

    private static int CountSetBits(ulong n) => BitOperations.PopCount(n);
    private static void Swap(ref Move a, ref Move b) => (a, b) = (b, a);
}