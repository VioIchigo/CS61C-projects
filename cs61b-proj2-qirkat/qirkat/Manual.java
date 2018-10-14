package qirkat;

import static qirkat.PieceColor.*;
import static qirkat.Command.Type.*;

/** A Player that receives its moves from its Game's getMoveCmnd method.
 *  @author Violet Fu and Skeleton
 */
class Manual extends Player {

    /** A Player that will play MYCOLOR on GAME, taking its moves from
     *  GAME. */
    Manual(Game game, PieceColor myColor) {
        super(game, myColor);
        _prompt = myColor + ": ";
    }

    @Override
    Move myMove() {
        assert myColor() == game().board().whoseMove();
        assert !game().board().gameOver();
        Command command = game().getMoveCmnd(_prompt);
        if (command == null) {
            return null;
        }
        Move mov = Move.parseMove(command.operands()[0]);
        if (board().legalMove(mov)) {
            return mov;
        } else {
            System.out.println("that move is illegal.");
            return null;
        }
    }

    /** Identifies the player serving as a source of input commands. */
    private String _prompt;
}

