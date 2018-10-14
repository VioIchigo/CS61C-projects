package qirkat;

import java.util.Observable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Formatter;
import java.util.Observer;

import static qirkat.PieceColor.*;
import static qirkat.Move.*;

/** A Qirkat board.   The squares are labeled by column (a char value between
 *  'a' and 'e') and row (a char value between '1' and '5'.
 *
 *  For some purposes, it is useful to refer to squares using a single
 *  integer, which we call its "linearized index".  This is simply the
 *  number of the square in row-major order (with row 0 being the bottom row)
 *  counting from 0).
 *
 *  Moves on this board are denoted by Moves.
 *  @author Violet Fu and Skeleton
 */
class Board extends Observable {

    /** A new, cleared board at the start of the game. */
    Board() {
        clear();
    }

    /** A copy of B. */
    Board(Board b) {
        internalCopy(b);
    }

    /** Return a constant view of me (allows any access method, but no
     *  method that modifies it). */
    Board constantView() {
        return this.new ConstantBoard();
    }

    /** Clear me to my starting state, with pieces in their initial
     *  positions. */
    void clear() {
        _whoseMove = WHITE;
        _gameOver = false;

        for (int i = 0; i <= MAX_INDEX; i += 1) {
            set(i, BLACK);
        }
        for (int i : _initWhite) {
            set(i, WHITE);
        }
        set(12, EMPTY);
        _history.clear();

        setInitialDirection(MAX_INDEX);

        setChanged();
        notifyObservers();
    }
    /** Copy B into me. */
    void copy(Board b) {
        internalCopy(b);
    }

    /** Copy B into me. */
    private void internalCopy(Board b) {
        for (int i = 0; i < SIDE * SIDE; i += 1) {
            set(i, b.get(i));
        }

        _directions = b._directions;
        _whoseMove = b._whoseMove;
        _history = b._history;
    }

    /** Set my contents as defined by STR.  STR consists of 25 characters,
     *  each of which is b, w, or -, optionally interspersed with whitespace.
     *  These give the contents of the Board in row-major order, starting
     *  with the bottom row (row 1) and left column (column a). All squares
     *  are initialized to allow horizontal movement in either direction.
     *  NEXTMOVE indicates whose move it is.
     */
    void setPieces(String str, PieceColor nextMove) {
        if (nextMove == EMPTY || nextMove == null) {
            throw new IllegalArgumentException("bad player color");
        }
        str = str.replaceAll("\\s", "");
        if (!str.matches("[bw-]{25}")) {
            throw new IllegalArgumentException("bad board description");
        }

        for (int k = 0; k < str.length(); k += 1) {
            switch (str.charAt(k)) {
            case '-':
                set(k, EMPTY);
                break;
            case 'b': case 'B':
                set(k, BLACK);
                break;
            case 'w': case 'W':
                set(k, WHITE);
                break;
            default:
                break;
            }
        }

        _whoseMove = nextMove;
        setInitialDirection(MAX_INDEX);

        _history.clear();
        setChanged();
        notifyObservers();
    }

    /** Return true iff the game is over: i.e., if the current player has
     *  no moves. */
    boolean gameOver() {
        return _gameOver;
    }

    /** Return the current contents of square C R, where 'a' <= C <= 'e',
     *  and '1' <= R <= '5'.  */
    PieceColor get(char c, char r) {
        assert validSquare(c, r);
        return get(index(c, r));
    }

    /** Set all the allowed directions to both directions.
     *  Takes in an int INDEX and returns nothing. */
    private void setInitialDirection(int index) {
        for (int i = 0; i <= index; i += 1) {
            ArrayList<String> init = new ArrayList<>();
            init.add("Both");
            _directions.put(i, init);
        }
    }

    /** Set the allowed moving direction at the given int
     *  INDEX to a string S and return nothing. */
    private void setDirection(int index, String s) {
        _directions.get(index).add(s);
    }


    /** Return the current contents of the square at linearized index K. */
    PieceColor get(int k) {
        assert validSquare(k);
        return _board[k];
    }

    /** Set get(C, R) to V, where 'a' <= C <= 'e', and
     *  '1' <= R <= '5'. */
    private void set(char c, char r, PieceColor v) {
        assert validSquare(c, r);
        set(index(c, r), v);
    }

    /** Set get(K) to V, where K is the linearized index of a square. */
    private void set(int k, PieceColor v) {
        assert validSquare(k);
        _board[k] = v;
    }

    /** Return true iff MOV is legal on the current board. */
    boolean legalMove(Move mov) {
        if (mov == null || !validSquare(mov.toIndex())
                || !validSquare(mov.fromIndex())) {
            throw new IllegalArgumentException("Illegal move");
        }
        PieceColor from = get(mov.fromIndex());
        PieceColor to = get(mov.toIndex());
        if (!mov.isJump() && jumpPossible()) {
            return false;
        }
        if (from != _whoseMove) {
            return false;
        } else if (mov.isJump()) {
            return checkJump(mov, false);
        } else if (from == BLACK && row(mov.fromIndex()) == '1') {
            return false;
        } else if (from == WHITE && row(mov.fromIndex()) == '5') {
            return false;
        } else if (mov.isLeftMove()) {
            ArrayList<String> rec = _directions.get(mov.fromIndex());
            return get(mov.toIndex()) == EMPTY
                    && !rec.get(rec.size() - 1).equals("Right");
        } else if (mov.isRightMove()) {
            ArrayList<String> rec = _directions.get(mov.fromIndex());
            return get(mov.toIndex()) == EMPTY
                    && !rec.get(rec.size() - 1).equals("Left");
        } else if (from == BLACK) {
            if (mov.fromIndex() % 2 == 0 && to == EMPTY) {
                return mov.fromIndex() - mov.toIndex() == SIDE
                        || mov.fromIndex() - mov.toIndex() == SIDE - 1
                        || mov.fromIndex() - mov.toIndex() == SIDE + 1;
            } else {
                return mov.fromIndex() - mov.toIndex() == SIDE && to == EMPTY;
            }
        } else if (from == WHITE) {
            if (mov.fromIndex() % 2 == 0 && to == EMPTY) {
                return mov.toIndex() - mov.fromIndex() == SIDE
                        || mov.toIndex() - mov.fromIndex() == SIDE + 1
                        || mov.toIndex() - mov.fromIndex() == SIDE - 1;
            } else {
                return mov.toIndex() - mov.fromIndex() == SIDE && to == EMPTY;
            }
        }
        return false;
    }



    /** Return a list of all legal moves from the current position. */
    ArrayList<Move> getMoves() {
        ArrayList<Move> result = new ArrayList<>();
        getMoves(result);
        return result;
    }

    /** Add all legal moves from the current position to MOVES. */
    void getMoves(ArrayList<Move> moves) {
        if (gameOver()) {
            return;
        }
        if (jumpPossible()) {
            for (int k = 0; k <= MAX_INDEX; k += 1) {
                getJumps(moves, k);
            }
        } else {
            for (int k = 0; k <= MAX_INDEX; k += 1) {
                getMoves(moves, k);
            }
        }
    }

    /** Add all legal non-capturing moves from the position
     *  with linearized index K to MOVES. */
    private void getMoves(ArrayList<Move> moves, int k) {
        for (int i = -1; i <= 1; i += 1) {
            for (int j = -1; j <= 1; j += 1) {
                char nextCol = (char) (col(k) + i);
                char nextRow = (char) (row(k) + j);
                if (validSquare(nextCol, nextRow)) {
                    Move rec = Move.move(col(k),
                            row(k), nextCol, nextRow);
                    if (legalMove(rec)) {
                        moves.add(rec);
                    }
                }
            }
        }
    }

    /** Add all legal captures from the position with linearized index K
     *  to MOVES. */
    private void getJumps(ArrayList<Move> moves, int k) {
        if (!jumpPossible(k)) {
            return;
        }

        for (int i = -2; i <= 2; i += 2) {
            for (int j = -2; j <= 2; j += 2) {
                char nextCol = (char) (col(k) + i);
                char nextRow = (char) (row(k) + j);
                if (validSquare(nextCol, nextRow)) {
                    Move rec = Move.move(col(k),
                            row(k), nextCol, nextRow);
                    if (checkJump(rec, true)) {
                        PieceColor middle = get(nextCol, nextRow);
                        set(nextCol, nextRow, get(k));
                        ArrayList<Move> record = new ArrayList<>();
                        getJumps(record, index(nextCol, nextRow));
                        if (record.size() == 0) {
                            record.add(null);
                        }
                        set(nextCol, nextRow, middle);
                        for (Move mov : record) {
                            Move jump = Move.move(rec, mov);
                            moves.add(jump);
                        }
                    }
                }
            }
        }
    }


    /** Takes in a move MOV and returns
     *  a boolean to check if it is a jump, not
     *  necessarily a complete one. */
    boolean partialJump(Move mov) {
        PieceColor init = get(mov.fromIndex());
        PieceColor to = get(mov.toIndex());
        if (mov.fromIndex() % 2 == 0) {
            return get(mov.jumpedCol(), mov.jumpedRow())
                    == init.opposite()
                    && to == EMPTY;
        } else {
            return index(mov.jumpedCol(), mov.jumpedRow()) % 2 == 0
                    && get(mov.jumpedCol(), mov.jumpedRow())
                    == init.opposite()
                    && to == EMPTY;
        }
    }


    /** Return true iff MOV is a valid jump sequence on the current board.
     *  MOV must be a jump or null.  If ALLOWPARTIAL, allow jumps that
     *  could be continued and are valid as far as they go.  */
    boolean checkJump(Move mov, boolean allowPartial) {
        if (!mov.isJump()) {
            return false;
        }
        PieceColor compare = get(mov.fromIndex()).opposite();
        PieceColor init = get(mov.fromIndex());
        ArrayList<Integer> record = new ArrayList<>();
        if (allowPartial) {
            return partialJump(mov);
        } else {
            Move rec = mov;
            while (rec != null) {
                PieceColor from = get(rec.fromIndex());
                PieceColor to = get(rec.toIndex());
                record.add(rec.toIndex());
                char mCol = rec.jumpedCol();
                char mRow = rec.jumpedRow();
                if (rec.fromIndex() % 2 == 0) {
                    set(rec.fromIndex(), init);
                    if (get(mCol, mRow)
                            == compare && to == EMPTY) {
                        set(rec.fromIndex(), from);
                        rec = rec.jumpTail();
                    } else {
                        set(rec.fromIndex(), from);
                        return false;
                    }
                } else {
                    set(rec.fromIndex(), init);
                    if (index(mCol, mRow) % 2 == 0
                            && get(mCol, mRow) == compare
                            && to == EMPTY) {
                        set(rec.fromIndex(), from);
                        rec = rec.jumpTail();
                    } else {
                        set(rec.fromIndex(), from);
                        return false;
                    }
                }
            }
            for (int i : record) {
                set(i, init);
            }
            if (jumpPossible(record.get(record.size() - 1))) {
                for (int i : record) {
                    set(i, EMPTY);
                }
                return false;
            }
            for (int i : record) {
                set(i, EMPTY);
            }
            return true;
        }
    }


    /** Return true iff a jump is possible for a piece at position C R. */
    boolean jumpPossible(char c, char r) {
        return jumpPossible(index(c, r));
    }

    /** Return true iff a jump is possible for a piece at position with
     *  linearized index K. */
    boolean jumpPossible(int k) {
        if (get(k) != _whoseMove) {
            return false;
        }
        for (int i = -2; i <= 2; i += 2) {
            for (int j = -2; j <= 2; j += 2) {
                char nextCol = (char) (col(k) + i);
                char nextRow = (char) (row(k) + j);
                if (k % 2 == 0) {
                    if (validSquare(nextCol, nextRow)) {
                        int mid = (k + index(nextCol, nextRow)) / 2;
                        if (get(nextCol, nextRow)
                                == EMPTY && get(mid) == get(k).opposite()) {
                            return true;
                        }
                    }
                } else {
                    if (i == 0 || j == 0) {
                        if (validSquare(nextCol, nextRow)) {
                            int mid = (k + index(nextCol, nextRow)) / 2;
                            if (get(nextCol, nextRow)
                                    == EMPTY && get(mid) == get(k).opposite()) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        return false;
    }

    /** Return true iff a jump is possible from the current board. */
    boolean jumpPossible() {
        for (int k = 0; k <= MAX_INDEX; k += 1) {
            if (jumpPossible(k)) {
                return true;
            }
        }
        return false;
    }

    /** Return the color of the player who has the next move.  The
     *  value is arbitrary if gameOver(). */
    PieceColor whoseMove() {
        return _whoseMove;
    }

    /** Perform the move C0R0-C1R1, or pass if C0 is '-'.  For moves
     *  other than pass, assumes that legalMove(C0, R0, C1, R1). */
    void makeMove(char c0, char r0, char c1, char r1) {
        makeMove(Move.move(c0, r0, c1, r1, null));
    }

    /** Make the multi-jump C0 R0-C1 R1..., where NEXT is C1R1....
     *  Assumes the result is legal. */
    void makeMove(char c0, char r0, char c1, char r1, Move next) {
        makeMove(Move.move(c0, r0, c1, r1, next));
    }

    /** Make the Move MOV on this Board, assuming it is legal. */
    void makeMove(Move mov) {
        assert legalMove(mov);
        if (mov.isJump()) {
            assert checkJump(mov, false);
            Move rec = mov;
            while (rec != null) {
                set(rec.toIndex(), _whoseMove);
                set(rec.fromIndex(), EMPTY);
                set(rec.jumpedCol(), rec.jumpedRow(), EMPTY);
                rec = rec.jumpTail();
            }
            setDirection(mov.fromIndex(), "Both");
        } else {
            int to = mov.toIndex();
            int from = mov.fromIndex();
            set(mov.toIndex(), _whoseMove);
            set(mov.fromIndex(), EMPTY);
            if (mov.isRightMove()) {
                setDirection(to,  "Right");
            } else if (mov.isLeftMove()) {
                setDirection(to,  "Left");
            } else {
                setDirection(to, "Both");
            }
            setDirection(from, "Both");
        }

        _history.add(mov);
        _whoseMove = _whoseMove.opposite();
        _gameOver = !isMove();
        setChanged();
        notifyObservers();
    }

    /** Undo the last move, if any. */
    void undo() {
        Move rec = _history.get(_history.size() - 1);
        int to = rec.fromIndex();
        _directions.get(to).remove(_directions.get(to).size() - 1);
        if (rec.isJump()) {
            Move middle = Move.move(rec.col1(), rec.row1(),
                    rec.col0(), rec.row0(), null);
            rec = rec.jumpTail();
            while (rec != null) {
                middle = Move.move(rec.col1(), rec.row1(),
                        rec.col0(), rec.row0(), middle);
                rec = rec.jumpTail();
            }
            while (middle != null) {
                set(middle.fromIndex(), EMPTY);
                set(middle.toIndex(), _whoseMove.opposite());
                set(middle.jumpedCol(), middle.jumpedRow(), _whoseMove);
                middle = middle.jumpTail();
            }

        } else {
            int from = rec.toIndex();
            set(to, _whoseMove.opposite());
            set(from, EMPTY);

            _directions.get(from).remove(_directions.get(from).size() - 1);
        }

        _whoseMove = _whoseMove.opposite();
        _history.remove(_history.size() - 1);
        setChanged();
        notifyObservers();
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof Board) {
            Board b = (Board) o;
            return (b.toString().equals(toString())
                    && _whoseMove == b.whoseMove()
                    && _directions == b._directions
                    && _history == b._history);
        } else {
            return false;
        }
    }

    @Override
    public int hashCode() {
        int h = 0;
        for (int i = 0; i <= MAX_INDEX; i += 1) {
            h = h * 10;
            PieceColor rec = get(i);
            if (rec == BLACK) {
                h += 1;
            } else if (rec == WHITE) {
                h += 2;
            } else {
                h += 3;
            }
        }
        return h;
    }

    @Override
    public String toString() {
        return toString(false);
    }


    /** Return a text depiction of the board.  If LEGEND, supply row and
     *  column numbers around the edges. */
    String toString(boolean legend) {
        Formatter out = new Formatter();
        if (!legend) {
            for (char i = '5'; i >= '1'; i = (char) (i - 1)) {
                out.format(" ");
                for (char j = 'a'; j <= 'e'; j = (char) (j + 1)) {
                    out.format(" ");
                    out.format("%s", get(j, i).shortName());
                }
                if (i != '1') {
                    out.format("%n");
                }
            }
        } else {
            if (legend) {
                int s = SIDE;
                for (char i = '5'; i >= '1'; i = (char) (i - 1)) {
                    out.format("%s ", i);
                    for (char j = 'a'; j <= 'e'; j = (char) (j + 1)) {
                        out.format(" ");
                        out.format("%s", get(j, i).shortName());
                    }
                    if (i != '1') {
                        out.format("%n");
                    }
                }
                out.format("   a b c d e");
            }
        }
        return out.toString();
    }

    /** Return true iff there is a move for the current player. */
    private boolean isMove() {
        return this.getMoves().size() != 0;
    }

    /** Player that is on move. */
    private PieceColor _whoseMove;

    /** Set true when game ends. */
    private boolean _gameOver;

    /** Record all the PieceColors in this board. */
    private PieceColor[] _board = new PieceColor[SIDE * SIDE];

    /** All indexes for the initial white spaces. */
    private static int[] _initWhite
            = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14};

    /** Records of all the directions. Used to prevent tie. */
    private HashMap<Integer, ArrayList<String>> _directions = new HashMap<>();

    /** The history of all the moves. */
    private MoveList _history = new MoveList();

    /** Convenience value giving values of pieces at each ordinal position. */
    static final PieceColor[] PIECE_VALUES = PieceColor.values();

    /** One cannot create arrays of ArrayList<Move>, so we introduce
     *  a specialized private list type for this purpose. */
    private static class MoveList extends ArrayList<Move> {
    }

    /** A read-only view of a Board. */
    private class ConstantBoard extends Board implements Observer {
        /** A constant view of this Board. */
        ConstantBoard() {
            super(Board.this);
            Board.this.addObserver(this);
        }

        @Override
        void copy(Board b) {
            assert false;
        }

        @Override
        void clear() {
            assert false;
        }

        @Override
        void makeMove(Move move) {
            assert false;
        }

        /** Undo the last move. */
        @Override
        void undo() {
            assert false;
        }

        @Override
        public void update(Observable obs, Object arg) {
            super.copy((Board) obs);
            setChanged();
            notifyObservers(arg);
        }
    }
}
