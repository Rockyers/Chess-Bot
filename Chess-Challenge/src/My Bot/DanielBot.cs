using ChessChallenge.API;

namespace Chess_Challenge.My_Bot;

public class DanielBot : IChessBot
{
    public Move Think(Board board, Timer timer)
    {
        Move[] moves = board.GetLegalMoves();
        return moves[0];
    }
}