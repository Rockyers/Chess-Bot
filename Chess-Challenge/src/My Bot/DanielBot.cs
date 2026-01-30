using System;
using System.Collections.Generic;
using System.Diagnostics;
using ChessChallenge.API;

namespace Chess_Challenge.My_Bot;

public class DanielBot : IChessBot
{
    private enum TransFlag
    {
        Exact,
        Lowerbound,
        Upperbound
    }

    private struct TransEntry
    {
        public TransFlag Flag;
        public int Depth;
        public int Eval;
        public Move BestMove;
    }

    private readonly Dictionary<ulong, TransEntry> _transpositionTable = new();

    private const int baseDepth = 5;
    private const int maxDepth = 32;
    public Move Think(Board board, Timer timer)
    {
        Console.WriteLine($"[Xendy] Starting thinking...");

        var moves = GetSortedLegalMoves(board);
        var bestMove = moves[0];
        var currentEval = 0;
        for (var iterativeDepth = 1; iterativeDepth < baseDepth; iterativeDepth++)
        {
            var alpha = int.MinValue;
            foreach (var move in moves)
            {
                board.MakeMove(move);
                var eval = -EvaluateRecursive(iterativeDepth, maxDepth,  1, board, timer, alpha, int.MaxValue);
                board.UndoMove(move);

                if (eval < alpha) continue;
                bestMove = move;
                alpha = eval;
            }
            currentEval = alpha;
        }
        Console.WriteLine($"[Xendy] Eval: {currentEval}, Trans: {_transpositionTable.Count} entries");
        return bestMove;
    }

    private int EvaluateRecursive(int depth, int maximumDepth, int currentDepth, Board board, Timer timer, int alpha, int beta)
    {
        var currentEval = GetEval(board);
        if (depth <= 0 || maximumDepth <= 0 || board.IsInCheckmate()) return currentEval;
        
        var hasTransposition = _transpositionTable.TryGetValue(board.ZobristKey, out var transEntry);
        if (hasTransposition && transEntry.Depth >= depth)
        {
            switch (transEntry.Flag)
            {
                case TransFlag.Lowerbound when transEntry.Eval >= beta:
                case TransFlag.Upperbound when transEntry.Eval <= alpha:
                case TransFlag.Exact:
                    return transEntry.Eval;
            }
        }
        var originalAlpha = alpha;
        // Get opponent's best move
        var bestEval = int.MinValue;
        var moves = GetSortedLegalMoves(board);
        var bestMove = moves[0];
        foreach (var move in moves)
        {
            board.MakeMove(move);
            var nextDepth = GetNextDepth(depth, move, board);
            // Reverse alpha and beta becauese it's the opponent's turn
            var eval = -EvaluateRecursive(nextDepth, maximumDepth - 1, currentDepth + 1, board, timer, -beta, -alpha);
            board.UndoMove(move);
            if (eval > alpha) alpha = eval;
            if (eval > bestEval)
            {
                bestEval = eval;
                bestMove = move;
            }
            if (alpha >= beta) break;
        }

        _transpositionTable[board.ZobristKey] = new TransEntry
        {
            Eval = bestEval,
            Depth = depth,
            Flag = bestEval <= originalAlpha ? TransFlag.Upperbound 
                : bestEval >= beta ? TransFlag.Lowerbound 
                : TransFlag.Exact,
            BestMove = bestMove
        };
        return bestEval;
    }

    // No clue if NonAllocc is actually faster since we sitll need to allocate stuff to stack
    private Move[] GetSortedLegalMoves(Board board)
    {
        var moves = board.GetLegalMoves();
        Move? hashMove = null;
    
        if (_transpositionTable.TryGetValue(board.ZobristKey, out var entry))
        {
            hashMove = entry.BestMove;
        }
        moves.Sort((a, b) =>
        {
            if (hashMove.HasValue)
            {
                if (a.Equals(hashMove.Value)) return -1000;
                if (b.Equals(hashMove.Value)) return 1000;
            }
            return GetMoveScore(b) - GetMoveScore(a);
        });
        return moves;
    }

    private int GetMoveScore(Move move)
    {
        if (move.IsCapture)
        {
            return _materialEval[(int)move.CapturePieceType] - _materialEval[(int)move.MovePieceType];
        }
        return 0;
    }

    private int GetNextDepth(int currentDepth, Move move, Board board)
    {
        var depth = currentDepth - 1;
        if (currentDepth != 0) return depth;
        
        // Extensions
        if (move.IsCapture || board.IsInCheck()) depth += 1;
        return depth;
    }

    private int GetEval(Board board)
    {
        if (board.IsInCheckmate()) return -20000000;
        var me = GetMaterial(board, board.IsWhiteToMove);
        var other = GetMaterial(board, !board.IsWhiteToMove);
        return me - other;
    }

    private readonly int[] _materialEval = [ 000, 100, 300, 310, 500, 800, 2000 ];
    private int GetMaterial(Board board, bool white)
    {
        var material = 0;
        for (var i = 1; i < 7; i++)
        {
            material += CountOnesBitwise(board.GetPieceBitboard((PieceType)i, white)) * _materialEval[i];
        }
        return material;
        
    }
    
    public static int CountOnesBitwise(ulong n)
    {
        int count = 0;
        while (n != 0)
        {
            n &= (n - 1); 
            count++;
        }
        return count;
    }
}