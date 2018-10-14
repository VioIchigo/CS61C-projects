/* Author: Paul N. Hilfinger.  (C) 2008. */

package qirkat;

import org.junit.Test;
import static org.junit.Assert.*;

import static qirkat.Move.*;

/** Test Move creation.
 *  @author Skeleton
 */
public class MoveTest {

    @Test
    public void testMove1() {
        Move m = move('a', '3', 'b', '2');
        assertNotNull(m);
        assertFalse("move should not be jump", m.isJump());
    }

    @Test
    public void testJump1() {
        Move m = move('a', '3', 'a', '5');
        assertNotNull(m);
        assertTrue("move should be jump", m.isJump());
    }

    @Test
    public void testString() {
        Move x = move('a', '3', 'a', '5',
                move('a', '5', 'c', '3'));
        Move y = move(move('a', '1', 'a', '3'), x);
        assertEquals("a1-a3-a5-c3", y.toString());
        assertEquals("a3-b2", move('a', '3', 'b', '2').toString());
        assertEquals("a3-a5", move('a', '3', 'a', '5').toString());
        assertEquals("a3-a5-c3", move('a', '3', 'a', '5',
                                      move('a', '5', 'c', '3')).toString());
        Move m = move('a', '3', 'b', '2');
        assertEquals("a3-b2", move(m, null).toString());
        assertEquals("a3-b2", move(null, m).toString());
    }

    @Test
    public void testParseString() {
        assertEquals("a3-b2", parseMove("a3-b2").toString());
        assertEquals("a3-a5", parseMove("a3-a5").toString());
        assertEquals("a3-a5-c3", parseMove("a3-a5-c3").toString());
        assertEquals("a3-a5-c3-e1", parseMove("a3-a5-c3-e1").toString());
    }

    /* More tests written by myself. */
    @Test
    public void testIsLeftMove() {
        Move m = move('a', '3', 'b', '2');
        assertFalse(m.isLeftMove());
        Move n = move('b', '3', 'a', '3');
        assertTrue(n.isLeftMove());
        Move o = move('c', '4', 'd', '4');
        assertFalse(o.isLeftMove());
    }

    @Test
    public void testIsRightMove() {
        Move m = move('a', '3', 'b', '2');
        assertFalse(m.isRightMove());
        Move n = move('c', '4', 'd', '4');
        assertTrue(n.isRightMove());
        Move l = move('b', '3', 'a', '3');
        assertFalse(l.isRightMove());
    }

    @Test
    public void testJump2() {
        Move l = move('b', '3', 'a', '3');
        assertFalse(l.isJump());
    }


    @Test
    public void testColumnsAndRows() {
        Move m = move('a', '3', 'b', '2');
        assertEquals('a', m.col0());
        assertEquals('3', m.row0());
        assertEquals('b', m.col1());
        assertEquals('2', m.row1());
    }



}
