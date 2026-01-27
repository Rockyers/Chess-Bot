using ChessChallenge.API;

public class EthanBot : IChessBot
{
    public Move Think(Board board, Timer timer)
    {
        Move[] moves = board.GetLegalMoves();
                                return moves[0];
    }
}