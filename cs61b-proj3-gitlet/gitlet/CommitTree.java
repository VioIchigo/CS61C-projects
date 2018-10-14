package gitlet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/** The message tree of all the commits we have committed.
 *  @author Violet Fu */
public class CommitTree extends Utils implements Serializable {
    /** The staging area of this commit tree. */
    private HashMap<String, Blob> stage = new HashMap<>();
    /** The last Commit, which the head pointer points to. */
    private Commit lastCommit;
    /** Name of the head pointer. */
    private String headPointer;
    /** The files that should be removed in the next commit. */
    private List<String> removeFiles = new ArrayList<>();
    /** ALl the branches in this commit tree. */
    private HashMap<String, Commit> branches = new HashMap<>();
    /** All the commits we have committed in this commit tree. */
    private ArrayList<Commit> commitRecord = new ArrayList<>();


    /** Constructor.*/
    public CommitTree() {
        lastCommit = new Commit();
        headPointer = "master";
        branches.put(headPointer, lastCommit);
        commitRecord.add(lastCommit);
    }

    /** Add a new file to this commit tree.
     *  It takes in a blob BLOB and returns nothing. */
    public void addFile(Blob blob) {
        String name = blob.getName();
        if (lastCommit == null
                || lastCommit.getParent() == null
                || lastCommit.getContent() == null) {
            stage.put(name, blob);
            return;
        }
        if (lastCommit.getContent().containsKey(name)) {
            String lastVersion = lastCommit.getContent().get(name).getSha1();
            String newVersion = blob.getSha1();
            if (lastVersion.equals(newVersion)) {
                if (stage.containsValue(blob)) {
                    stage.remove(blob.getName());
                }
            } else {
                stage.put(name, blob);
            }
        } else {
            stage.put(name, blob);
        }
        if (removeFiles.contains(name)) {
            removeFiles.remove(name);
        }
    }

    /** Add a commit COMMIT to this commit Tree. */
    public void addCommit(Commit commit) {
        lastCommit = commit;
        for (String s : removeFiles) {
            lastCommit.removeContent(s);
        }
        for (String name : lastCommit.getContent().keySet()) {
            if (stage.containsKey(name)) {
                lastCommit.getContent().remove(name);
            }
        }
        for (Blob blob : stage.values()) {
            lastCommit.getContent().put(blob.getName(), blob);
        }
        branches.put(headPointer, lastCommit);
        lastCommit.setSHA1();
        stage.clear();
        commitRecord.add(lastCommit);
        removeFiles.clear();
    }


    /** This method is used to find the commit with the given string
     *  MSG. It returns true if this commit exists in
     *  LIST and false otherwise. */
    public boolean findWithMessage(String msg, ArrayList<String> list) {
        boolean result = false;
        for (Commit commit : commitRecord) {
            if (commit.getMessage().equals(msg)) {
                result = true;
                list.add(commit.getSha1());
            }
        }
        return result;
    }


    /** Remove a file whose name is NAME from this commit tree. */
    public void removeFile(String name) {
        if (stage.containsKey(name)) {
            stage.remove(name);
        }
        if (lastCommit.getContent().containsKey(name)) {
            removeFiles.add(name);
        }
    }


    /** return the last commit. */
    public Commit getLastCommit() {
        return lastCommit;
    }
    /** set the last commit to COMMIT. */
    public void setLastCommit(Commit commit) {
        this.lastCommit = commit;
    }
    /** remove a file whose name is NAME from the staging area. */
    public void stageRemove(String name) {
        stage.remove(name);
    }
    /** return the content of the staging area of this commit tree. */
    public HashMap<String, Blob> getStage() {
        return stage;
    }
    /** clear the staging area of this commit tree. */
    public void clearStage() {
        stage.clear();
    }
    /** add a new file whose name is NAME and whose blob is BLOB
     *  to the staging area. */
    public void stageAdd(String name, Blob blob) {
        stage.put(name, blob);
    }
    /** return the list of files that should be removed. */
    public List<String> getRemoveFiles() {
        return removeFiles;
    }
    /** return the head pointer of this commit. */
    public String getHeadPointer() {
        return headPointer;
    }
    /** return the commit record. */
    public ArrayList<Commit> getCommitRecord() {
        return commitRecord;
    }
    /** return the branches of this commit tree. */
    public HashMap<String, Commit> getBranches() {
        return branches;
    }
    /** set a pointer to a branch whose name is NAME to COMMIT. */
    public void setBranches(String name, Commit commit) {
        branches.put(name, commit);
    }
    /** remove a branch BRANCH. */
    public void removeBranch(String branch) {
        branches.remove(branch);
    }
    /** add a new file whose name is NAME to the remove list. */
    public void setRemoveFiles(String name) {
        removeFiles.add(name);
    }
    /** change the head pointer to POINTER. */
    public void setHeadPointer(String pointer) {
        headPointer = pointer;
    }


}
