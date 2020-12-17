package qirkat;

import org.junit.Test;

import java.util.ArrayList;

import static org.junit.Assert.*;
import static qirkat.Move.move;
import static qirkat.PieceColor.*;

/** Tests of the Board class.
 *  @author Violet Fu and Skeleton
 */
public class BoardTest {

    private static final String INIT_BOARD =
        "  b b b b b\n  b b b b b\n  b b - w w\n  w w w w w\n  w w w w w";

    private static final String[] GAME1 =
    { "c2-c3", "c4-c2",
      "c1-c3", "a3-c1",
      "c3-a3", "c5-c4",
      "a3-c5-c3",
    };


    private static final String BOARDREPR =
            "  b b b b b\n  b b w b b\n  b b - w w\n  w w w w w\n  b w w w w";

    private static final String BOARD_REPR_2 =
            "  w - - w -\n  w b w - -\n  b - - b -\n  - - w b -\n  b b w b w";

    private static final String BOARD_REPR_3 =
            "  - - - - -\n  - - - - -\n  - w - - -\n  - - b - -\n  - - - - -";

    private static final String BOARD_REPR_4 =
            "  - - - - -\n  - b - w w\n  - - b b b\n  b b - - -\n  w - w - -";

    private static final String BOARD_REPR_5 =
            "  - - b - -\n  - - b - -\n  - b - - -\n  b - - b -\n  w - - - -";

    private static final String BOARD_REPR_6 =
            "  w w - w w\n  w w - w w\n  b b w w w\n  b b - b b\n  b b b b b";

    private static final String BOARD_REPR_7 =
            "  w w - w w\n  - - - w w\n  - - - - w\n  b - - b b\n b w b b -";

    private static final String BOARD_REPR_8 =
            "  b - b - -\n  - - - - w\n  - - - w -\n  - - - - w\n  - - - - w";
    private static final String BOARD_REPR_9 =
            "  w w b w w\n  w - - w w\n  w b - w w\n  b - - b b\n  b b b b b";

    private static final String BOARD_REPR_10 =
            "  b - - - -\n  - - - - -\n  w - b - -\n  - - - - -\n  w - - - w";



    private static final String GAME1_BOARD =
        "  b b - b b\n  b - - b b\n  - - w w w\n  w - - w w\n  w w b w w";

    private static void makeMoves(Board b, String[] moves) {
        for (String s : moves) {
            b.makeMove(Move.parseMove(s));
        }
    }

    @Test
    public void testInit1() {
        Board b0 = new Board();
        assertEquals(INIT_BOARD, b0.toString());
    }

    @Test
    public void testMoves1() {
        Board b0 = new Board();
        makeMoves(b0, GAME1);
        assertEquals(GAME1_BOARD, b0.toString());
    }


    @Test
    public void testUndo() {
        Board b0 = new Board();
        Board b1 = new Board(b0);
        makeMoves(b0, GAME1);
        Board b2 = new Board(b0);
        for (int i = 0; i < GAME1.length; i += 1) {
            b0.undo();
        }
        assertEquals("failed to return to start", b1, b0);
        makeMoves(b0, GAME1);
        assertEquals("second pass failed to reach same position", b2, b0);
    }

    /* My portion of the test. */
    @Test
    public void testGet() {
        Board b0 = new Board();
        assertEquals(BLACK, b0.get(23));
        assertEquals(WHITE, b0.get(3));
        assertEquals(EMPTY, b0.get(12));
        assertEquals(EMPTY, b0.get('c', '3'));
        assertEquals(WHITE, b0.get('a', '1'));
        assertEquals(BLACK, b0.get('d', '4'));
    }

    @Test
    public void testJumpPossible() {
        Board b = new Board();
        assertFalse(b.jumpPossible(7));
        assertFalse(b.jumpPossible(2));
        assertFalse(b.jumpPossible(22));
        assertFalse(b.jumpPossible(18));
        b.setPieces(BOARD_REPR_2, WHITE);
        assertTrue(b.jumpPossible(22));
        assertTrue(b.jumpPossible(17));
        assertFalse(b.jumpPossible(12));
        assertTrue(b.jumpPossible(5));
        assertFalse(b.jumpPossible(10));
        assertFalse(b.jumpPossible(6));
        b.setPieces(BOARD_REPR_3, BLACK);
        assertFalse(b.jumpPossible(17));
    }


    @Test
    public void testCopy() {
        Board b0 = new Board();
        b0.setPieces(BOARDREPR, BLACK);
        Board b1 = new Board();
        b1.copy(b0);
        assertTrue(b1.jumpPossible(2));
    }

    @Test
    public void testSetPieces() {
        Board b0 = new Board();
        b0.setPieces(BOARDREPR, BLACK);
        assertTrue(b0.jumpPossible(2));
    }

    @Test
    public void testClear() {
        Board b0 = new Board();
        b0.setPieces(BOARDREPR, BLACK);
        b0.clear();
        Board b1 = new Board();
        assertEquals(b1.toString(), b0.toString());
    }

    @Test
    public void testLegalMove() {
        Board b0 = new Board();
        b0.setPieces(BOARD_REPR_4, WHITE);
        Move m = move('e', '2', 'e', '3');
        assertFalse(b0.legalMove(m));
        Move n = move('e', '2', 'e', '1');
        assertFalse(b0.legalMove(n));
        Move p = move('c', '5', 'a', '3');
        assertFalse(b0.legalMove(p));
        Move q = move('e', '2', 'd', '3');
        assertFalse(b0.legalMove(q));
        Move o = move('e', '2', 'e', '4');
        assertTrue(b0.legalMove(o));
        Move b = move('b', '4', 'b', '3');
        assertFalse(b0.legalMove(b));
        Move a = move('c', '5', 'd', '4');
        assertFalse(b0.legalMove(a));
    }


    @Test
    public void testCheckJump() {
        Board b0 = new Board();
        b0.setPieces(BOARD_REPR_5, WHITE);
        Move move1 = move('a', '5', 'a', '3');
        Move move2 = move('a', '3', 'c', '3');
        Move move3 = move('c', '3', 'e', '5');
        Move move4 = move('c', '3', 'c', '1');
        assertTrue(b0.checkJump(move1, true));
        assertTrue(b0.checkJump(move(move1, move(move2, move3)), false));
        assertFalse(b0.checkJump(move(move1, move(move2, move4)), false));
        assertTrue(b0.checkJump(move(move1, move(move2, move4)), true));
    }

    @Test
    public void testMyConfusion() {
        Board b0 = new Board();
        b0.setPieces(BOARD_REPR_6, BLACK);
        Move move1 = move('d', '4', 'c', '2');
        assertFalse(b0.legalMove(move1));
        Board b1 = new Board();
        b1.setPieces(BOARD_REPR_7, WHITE);
        Move move2 = move('e', '3', 'e', '5');
        assertFalse(b1.legalMove(move2));
        Board b2 = new Board();
        b2.setPieces(BOARDREPR, WHITE);
        Move move3 = move('c', '2', 'c', '3');
        b2.makeMove(move3);
        ArrayList<Move> m = new ArrayList<>();
        b2.getMoves(m);
        System.out.println(m);
        Board b3 = new Board();
        b3.setPieces(BOARD_REPR_8, BLACK);
        Move move4 = move('a', '1', 'b', '1');
        System.out.println(b3.legalMove(move4));
        System.out.println(b3.getMoves());
        Board b4 = new Board();
        b4.setPieces(BOARD_REPR_9, WHITE);
        System.out.println(b4.getMoves());
        Board b5 = new Board();
        b5.setPieces(BOARD_REPR_10, BLACK);
        ArrayList<Move> n = new ArrayList<>();
        b5.getMoves(n);
        Move move5 = move('c', '3', 'd', '3');
        System.out.println(b5.legalMove(move5));
        System.out.println(n);
    }



}
