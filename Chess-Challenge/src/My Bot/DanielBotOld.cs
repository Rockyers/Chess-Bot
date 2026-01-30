using System;
using ChessChallenge.API;

namespace Chess_Challenge.My_Bot;

public class DanielBotOld : IChessBot
{
    
    int[] fileValues = [100, 200, 300, 400, 400, 300, 200, 100];
    int[] pawnFileValues = [-200, -200, 100, 400, 400, 100, -200, -200];
    // Piece values: null, pawn, knight, bishop, rook, queen, king
    int[] pieceValues = [000, 100, 200, 250, 300, 600, 2000];
    
    
    int[] pawnForwards = [000, 000, 100, 400, 410, 100, 200, 2000, 200000];

    private bool amIWhite = false;
    public Move Think(Board board, Timer timer)
    {
        amIWhite = board.IsWhiteToMove;
        var moves = board.GetLegalMoves();
        Random rng = new();
        var moveToPlay = moves[rng.Next(moves.Length)];
        var moveToPlayEval = 0;
        foreach (var move in moves)
        {
            if (MoveIsCheckmate(board, move))
            {
                return move;
            }

            var forwards = (amIWhite ? move.TargetSquare.Rank : 8 - move.TargetSquare.Rank);
            var eval = 0;
            switch (move.MovePieceType)
            {
                case PieceType.Pawn:
                    eval = pawnFileValues[move.TargetSquare.File];
                    eval += GetCaptureEval(board, move);
                    eval += pawnForwards[forwards];
                    break;
                case PieceType.Bishop:
                    eval += GetCaptureEval(board, move);
                    eval += 400;
                    break;
                case PieceType.Knight:
                    eval += GetCaptureEval(board, move);
                    eval += 500;
                    break;
                case PieceType.Rook:
                    eval += GetCaptureEval(board, move);
                    eval += 100;
                    break;
                case PieceType.Queen:
                    eval += GetCaptureEval(board, move);
                    break;
                case PieceType.King:
                    eval -= 200;
                    break;
            }

            eval += GetThreatEval(board, move);
            if (eval > moveToPlayEval)
            {
                moveToPlayEval = eval;
                moveToPlay = move;
            }
        }

        return moveToPlay;
    }
    
    bool MoveIsCheckmate(Board board, Move move)
    {
        board.MakeMove(move);
        bool isMate = board.IsInCheckmate();
        board.UndoMove(move);
        return isMate;
    }

    int GetCaptureEval(Board board, Move move)
    {
        var currentPieceValue = pieceValues[(int)move.MovePieceType];
        var capturePieceValue = pieceValues[(int)move.CapturePieceType];
        board.MakeMove(move);
        var moves = board.GetLegalMoves(true);
        var captureBack = false;
        var highestCaptureValue = 0;
        foreach (var capture in moves)
        {
            if (capture.TargetSquare == move.TargetSquare) captureBack = true;
            var materialDiff = GetMaterialEval(board);
            highestCaptureValue = Math.Min(highestCaptureValue, materialDiff);
        }
        
        board.UndoMove(move);
        return (captureBack ? capturePieceValue - currentPieceValue : currentPieceValue) + highestCaptureValue;
    }


    int GetThreatEval(Board board, Move move)
    {
        board.MakeMove(move);
        board.ForceSkipTurn();
        var moves = board.GetLegalMoves();
        var lowestCaptureEval = 0;
        var squareCanMove = 0;
        foreach (var capture in moves)
        {
            var captureEval = GetCaptureEval(board, capture);
            if (captureEval > 0) squareCanMove++;
            
            if (!move.IsCapture) continue;
            lowestCaptureEval = Math.Min(lowestCaptureEval, captureEval);
        }
        board.UndoSkipTurn();
        board.UndoMove(move);
        return lowestCaptureEval + squareCanMove;
    }

    int[] _materialEval = { 000, 100, 300, 310, 500, 800, 2000 };
    int GetMaterialEval(Board board)
    {
        int GetEval(bool w)
        {
            var eval = 0;
            for (var i = 1; i < 7; i++)
            {
                eval += CountOnesBitwise(board.GetPieceBitboard((PieceType)i, w)) * _materialEval[i];
            }
            return eval;
        }

        var whiteEval = GetEval(true);
        var blackEval = GetEval(false);

        return amIWhite ? whiteEval - blackEval : blackEval - whiteEval;
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