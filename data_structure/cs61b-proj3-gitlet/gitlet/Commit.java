package gitlet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;

import static gitlet.Utils.sha1;


/** A single Commit.
 *  @author Violet Fu */
public class Commit implements Serializable {

    /** UID of this class. */
    private static final long serialVersionUID = -2907363516942576948L;
    /** all blobs in this commit. */
    private HashMap<String, Blob> content = new HashMap<>();
    /** The commit message of this commit. */
    private String message;
    /** The timestamp of this commit. */
    private Date date;
    /** The parent of this commit. */
    private Commit parent;
    /** The second parent for a merged commit. */
    private Commit secondParent;

    /** sha1 code which can be used to
     *  distinguish between blob and message. */
    private String sha1;

    /** Constructor.*/
    public Commit() {
        parent = null;
        secondParent = null;
        message = "initial commit";
        date = new Date(0);
        setSHA1();
    }

    /** Constructor that takes in a CommitTree COMMITTREE, a
     *  commit COMMITPARENT, a date TIMESTAMP, a string COMMITMESSAGE.*/
    public Commit(CommitTree commitTree, Commit commitParent,
                  Date timestamp, String commitMessage) {
        for (HashMap.Entry<String, Blob> entry
                : commitParent.content.entrySet()) {
            String name = entry.getKey();
            Blob blob = entry.getValue();
            content.put(name, blob);
        }
        this.message = commitMessage;
        this.date = timestamp;
        this.parent = commitParent;
        this.secondParent = null;
    }

    /** The method to set the SHA1 code of this commit.
     *  it takes in nothing and returns nothing.
     *  the SHA1 code of the commit is determined by its
     *  timestamp, content, and its parent. */
    public void setSHA1() {
        if (parent == null) {
            List<Object> rec = new ArrayList<>();
            rec.add(date.toString());
            sha1 = sha1(rec);
        } else {
            List<Object> rec = new ArrayList<>();
            rec.add(date.toString());
            for (Blob blob : content.values()) {
                rec.add(blob.getBytes());
            }
            rec.add(parent.getSha1());
            sha1 = sha1(rec);
        }
    }

    /** This method is used to find all the ancestors of this
     *  commit and it is used for log command. It takes in
     *  nothing and returns an ArrayList.*/
    public ArrayList<Commit> findAncestors() {
        ArrayList<Commit> ancestors = new ArrayList<>();
        Commit rec = this;
        while (rec != null) {
            ancestors.add(rec);
            rec = rec.parent;
        }
        return ancestors;
    }

    /** Used to get the second parent of this commit.
     *  It takes in a commit COMMIT and returns nothing. */
    public void setSecondParent(Commit commit) {
        this.secondParent = commit;
    }
    /** Used to get the commit message of this commit.
     *  It takes in nothing and returns a string. */
    public String getMessage() {
        return this.message;
    }
    /** Used to get the sha1 code of this commit.
     *  It takes in nothing and returns a string. */
    public String getSha1() {
        return sha1;
    }
    /** Used to get the parent of this commit.
     *  It takes in nothing and returns a commit. */
    public Commit getParent() {
        return parent;
    }
    /** This method is used to change the parent of this commit.
     *  It takes in a commit COMMIT and returns nothing*/
    public void setParent(Commit commit) {
        parent = commit;
    }
    /** This method is used to get the timestamp of this commit.
     *  It takes in nothing and return a date*/
    public Date getDate() {
        return date;
    }
    /** This method is used to get the second parent of this commit.
     *  It takes in nothing and returns a commit. */
    public Commit getSecondParent() {
        return secondParent;
    }
    /** This method is used to get the content of this commit.
     *  It takes in nothing and returns a HashMap. */
    public HashMap<String, Blob> getContent() {
        return content;
    }
    /** This method is used to remove a file from the current commit.
     *  It takes in a string NAME and returns nothing. */
    public void removeContent(String name) {
        content.remove(name);
    }
}

