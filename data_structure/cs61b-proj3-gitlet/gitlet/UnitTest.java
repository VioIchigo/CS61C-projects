package gitlet;

import ucb.junit.textui;
import org.junit.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;

import static org.junit.Assert.*;

/** The suite of all JUnit tests for the gitlet package.
 *  @author Violet Fu
 */
public class UnitTest {

    /** Run the JUnit tests in the loa package. Add xxxTest.class entries to
     *  the arguments of runClasses to run other JUnit tests. */
    public static void main(String[] ignored) {
        textui.runClasses(UnitTest.class);
    }

    /** A dummy test to avoid complaint. */
    @Test
    public void placeholderTest() {

    }

    @Test
    public void testFind() {
        CommitTree commitTree1 = new CommitTree();
        Commit commit1 = new Commit(commitTree1,
                commitTree1.getLastCommit(), new Date(), "commit1");
        commitTree1.addCommit(commit1);
        Commit commit2 = new Commit(commitTree1,
                commitTree1.getLastCommit(), new Date(), "commit2");
        commitTree1.addCommit(commit2);
        Commit commitTwo = new Commit(commitTree1,
                commitTree1.getLastCommit(), new Date(), "commit2");
        commitTree1.addCommit(commit2);
        ArrayList<String> id = new ArrayList<>();
        boolean exist = commitTree1.findWithMessage("commit2", id);
        assertTrue(exist);
        assertTrue(id.size() == 2);
    }

    @Test
    public void testAdd() {
        CommitTree commitTree1 = new CommitTree();
        Commit commit1 = new Commit(commitTree1,
                commitTree1.getLastCommit(), new Date(), "commit1");
        commitTree1.addCommit(commit1);
        Commit commit2 = new Commit(commitTree1,
                commitTree1.getLastCommit(), new Date(), "commit2");
        commitTree1.addCommit(commit2);
        assertEquals(3, commitTree1.getCommitRecord().size());
        assertEquals(commit2, commitTree1.getLastCommit());
    }

    @Test
    public void testNewBlob() {
        List<String> files = Utils.plainFilenamesIn(".");
        for (String name : files) {
            File file = new File(name);
            String oldVersion = new Blob(file).getSha1();
            String newVersion = new Blob(file).getSha1();
            assertEquals(oldVersion, newVersion);
        }
    }



    @Test
    public void testSHA1() {
        CommitTree commitTree1 = new CommitTree();
        Commit commit1 = new Commit(commitTree1,
                commitTree1.getLastCommit(), new Date(), "commit1");
        commitTree1.addCommit(commit1);
        Commit commit2 = new Commit(commitTree1,
                commitTree1.getLastCommit(), new Date(), "commit2");
        commitTree1.addCommit(commit2);
        assertFalse(commit2.getSha1().equals(commit1.getSha1()));
    }

    @Test
    public void testFindAncestors() {
        CommitTree commitTree1 = new CommitTree();
        Commit commit1 = new Commit(commitTree1,
                commitTree1.getLastCommit(), new Date(), "commit1");
        commitTree1.addCommit(commit1);
        Commit commit2 = new Commit(commitTree1,
                commitTree1.getLastCommit(), new Date(), "commit2");
        commitTree1.addCommit(commit2);
        Commit commit3 = new Commit(commitTree1,
                commitTree1.getLastCommit(), new Date(), "commit3");
        commitTree1.addCommit(commit3);
        assertEquals(4, commit3.findAncestors().size());
    }


    @Test
    public void tryTest() {
        HashMap<String, String> hm = new HashMap<>();
        assertFalse(hm.equals(null));
        assertTrue(hm.size() == 0);
        String a = "a";
        hm.put("b", a);
        assertTrue(hm.containsValue(a));
        String bla = "blablablabla";
        assertTrue(bla.substring(0, 7).length() == 7);

    }


    @Test
    public void testPlainFilenamesIn() {
        File temp = new File(".gitlet");
        File[] temp1 = temp.listFiles();
        System.out.println(Utils.plainFilenamesIn("."));
        assertTrue(Utils.plainFilenamesIn(".").size() != 0);
    }

}


