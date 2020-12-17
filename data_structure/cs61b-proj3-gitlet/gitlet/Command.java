package gitlet;

import java.text.SimpleDateFormat;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;
import java.util.List;
import java.util.HashMap;
import java.util.Collections;

/** All the command we can use.
 *  @author Violet Fu */
public class Command extends Utils {

    /** Constructor.*/
    public Command() {
    }
    /** date format.*/
    private static SimpleDateFormat sdf
            = new SimpleDateFormat("EEE MMM d HH:mm:ss yyyy Z");

    /** Init method.*/
    public static void init() {
        File gitlet = new File(".gitlet");
        if (gitlet.exists()) {
            System.out.println("A Gitlet version-control"
                    + "system already exists in the current directory.");
            return;
        }
        try {
            CommitTree commitTree = new CommitTree();
            gitlet.createNewFile();
            ObjectOutputStream out = new
                    ObjectOutputStream(new FileOutputStream(gitlet));
            out.writeObject(commitTree);
            out.close();
        } catch (IOException e) {
            return;
        }
    }

    /** Add a file named NAME.*/
    public static void add(String name) throws GitletException {
        File gitlet = new File(".gitlet");
        CommitTree commitTree = readObject(gitlet, CommitTree.class);
        File file = new File(name);
        if (!file.exists()) {
            System.out.println("File does not exist.");
        } else {
            Blob blob = new Blob(file);
            commitTree.addFile(blob);
            writeObject(gitlet, commitTree);
        }
    }



    /** Make s new commit with commit message MSG.*/
    public static void commit(String msg) {
        File gitlet = new File(".gitlet");
        CommitTree commitTree = readObject(gitlet, CommitTree.class);
        if (commitTree.getStage().size() == 0
                && commitTree.getRemoveFiles().size() == 0) {
            System.out.println("No changes added to the commit.");
            return;
        }
        Date date = new Date();
        Commit newCommit = new Commit(commitTree,
                commitTree.getLastCommit(), date, msg);
        commitTree.addCommit(newCommit);
        writeObject(gitlet, commitTree);
    }

    /** Remove a file named MSG.*/
    public static void remove(String msg) {
        File gitlet = new File(".gitlet");
        CommitTree commitTree = readObject(gitlet, CommitTree.class);
        File file = new File(msg);
        if (!commitTree.getLastCommit().getContent().containsKey(msg)
                && !commitTree.getStage().containsKey(msg)) {
            System.out.println("No reason to remove the file.");
        } else {
            commitTree.removeFile(msg);
            Commit last = commitTree.getLastCommit();
            if (last != null || last.getContent().size() != 0) {
                if (commitTree.getLastCommit().getContent().containsKey(msg)
                        || commitTree.getStage().containsKey(msg)) {
                    file.delete();
                }
            }
        }
        writeObject(gitlet, commitTree);
    }


    /** log method.*/
    public static void log() {
        File gitlet = new File(".gitlet");
        CommitTree commitTree = readObject(gitlet, CommitTree.class);
        Commit pointer = commitTree.getLastCommit();
        while (pointer != null) {
            System.out.println("===");
            System.out.println("commit " + pointer.getSha1());
            if (pointer.getSecondParent() != null) {
                String msg1 = pointer.getParent().getSha1().substring(0, 7);
                String msg2 = pointer.getSecondParent().
                        getSha1().substring(0, 7);
                System.out.println("Merge: " + msg1 + " " + msg2);
            }
            System.out.println("Date: " + sdf.format(pointer.getDate()));
            System.out.println(pointer.getMessage());
            System.out.println();
            pointer = pointer.getParent();
        }
    }

    /** global log method.*/
    public static void globalLog() {
        File gitlet = new File(".gitlet");
        CommitTree commitTree = readObject(gitlet, CommitTree.class);
        for (Commit pointer : commitTree.getCommitRecord()) {
            System.out.println("===");
            System.out.println("commit " + pointer.getSha1());
            if (pointer.getSecondParent() != null) {
                String msg1 = pointer.getParent().getSha1().substring(0, 7);
                String msg2
                        = pointer.getSecondParent().getSha1().substring(0, 7);
                System.out.println("Merge: " + msg1 + " " + msg2);
            }
            System.out.println("Date: " + sdf.format(pointer.getDate()));
            System.out.println(pointer.getMessage());
            System.out.println();
        }
    }

    /** Find the commit with commit message MSG.*/
    public static void find(String msg) {
        File gitlet = new File(".gitlet");
        CommitTree commitTree = readObject(gitlet, CommitTree.class);
        ArrayList<String> id = new ArrayList<>();
        boolean exist = commitTree.findWithMessage(msg, id);
        if (!exist) {
            System.out.println("Found no commit with that message.");
        } else {
            for (String i : id) {
                System.out.println(i);
            }
        }
    }

    /** status method.*/
    public static void status() {
        File gitlet = new File(".gitlet");
        CommitTree commitTree = readObject(gitlet, CommitTree.class);
        List<String> allFiles = plainFilenamesIn(".");


        System.out.println("=== Branches ===");
        Collection<String> keys = commitTree.getBranches().keySet();
        List<String> list = new ArrayList<>(keys);
        Collections.sort(list);

        for (String branch : list) {
            if (branch.equals(commitTree.getHeadPointer())) {
                System.out.println("*" + branch);
            } else {
                System.out.println(branch);
            }
        }
        System.out.println();

        System.out.println("=== Staged Files ===");
        Collection<String> keySet = commitTree.getStage().keySet();
        List<String> lst = new ArrayList<>(keySet);
        Collections.sort(lst);

        for (String file : lst) {
            System.out.println(file);
        }
        System.out.println();

        System.out.println("=== Removed Files ===");
        List<String> remove = commitTree.getRemoveFiles();
        Collections.sort(remove);
        for (String file : commitTree.getRemoveFiles()) {
            System.out.println(file);
        }
        System.out.println();


        System.out.println("=== Modifications Not Staged For Commit ===");
        ArrayList<String> qualified = modificationsNoteStaged
                (commitTree, commitTree.getLastCommit(), allFiles);
        Collections.sort(qualified);
        for (String file : qualified) {
            System.out.println(file);
        }
        System.out.println();

        System.out.println("=== Untracked Files ===");
        ArrayList<String> qualified2 = untracedFiles
                (commitTree, commitTree.getLastCommit(), allFiles);
        Collections.sort(qualified2);
        for (String file : qualified2) {
            if (!file.equals(".gitlet")) {
                System.out.println(file);
            }
        }
        System.out.println();
    }




    /** Used to find the files that are present in the working directory
     *  but neither staged for addition nor tracked in in CURRENT in COMMITTREE
     *  in this commit COMMIT. Return an ArrayLis. */
    public static ArrayList<String> untracedFiles(
            CommitTree commitTree, Commit commit, List<String> current) {
        ArrayList<String> record = new ArrayList<>();
        for (String name : current) {
            if (!commitTree.getStage().containsKey(name)
                    && !commit.getContent().containsKey(name)) {
                record.add(name);
            }
        }
        return record;
    }


    /** Find the modifications in COMMITTREE after the last
     *  COMMIT not staged for commit.
     *  We need to compare all the files with those in CURRENT.
     *  Return an ArrayLis. */
    public static ArrayList<String> modificationsNoteStaged(
            CommitTree commitTree, Commit commit, List<String> current) {
        ArrayList<String> record = new ArrayList<>();
        HashMap<String, Blob> last = commit.getContent();

        for (HashMap.Entry<String, Blob> entry
                : last.entrySet()) {
            String blobName = entry.getKey();
            Blob blob = entry.getValue();
            if (current.contains(blobName)) {
                File file = new File(blobName);
                String newVersion = new Blob(file).getSha1();
                String oldVersion = blob.getSha1();
                if (!oldVersion.equals(newVersion)
                        && !commitTree.getStage().containsKey(blobName)) {
                    record.add(blobName + " (modified)");
                } else if (commitTree.getStage().containsKey(blobName)
                        && !commitTree.getStage().get(blobName).
                        getSha1().equals(newVersion)) {
                    record.add(blobName + " (modified)");
                }
            } else if (!commitTree.getRemoveFiles().contains(blobName)
                    && last.containsKey(blobName)
                    && !current.contains(blobName)) {
                record.add(blobName + " (deleted)");
            } else if (!current.contains(blobName)
                    && commitTree.getStage().containsKey(blobName)) {
                record.add(blobName + " (deleted)");
            }
        }

        return record;
    }



    /** make a new BRANCH.*/
    public static void branch(String branch) {
        File gitlet = new File(".gitlet");
        CommitTree commitTree = readObject(gitlet, CommitTree.class);
        if (commitTree.getBranches().containsKey(branch)) {
            System.out.println("A branch with that name already exists.");
        } else {
            commitTree.getBranches().put(branch, commitTree.getLastCommit());
        }
        writeObject(gitlet, commitTree);
    }




    /** checkout the file from the last commit whose name
     *  is NAME.*/
    public static void checkoutFileV1(String name) {
        File gitlet = new File(".gitlet");
        CommitTree commitTree = readObject(gitlet, CommitTree.class);
        Commit lastCommit = commitTree.getLastCommit();
        if (!lastCommit.getContent().containsKey(name)) {
            System.out.println("File does not exist in that commit.");
            return;
        }
        Blob blob = lastCommit.getContent().get(name);
        writeContents(new File(name), blob.getBytes());
        writeObject(gitlet, commitTree);
    }

    /** overwrite the file with FILENAME by FILE.*/
    public static void overwrite(String fileName, byte[] file) {
        File temp = new File(fileName);
        try {
            temp.createNewFile();
        } catch (IOException e) {
            System.out.println("error");
        }
        writeContents(temp, file);
    }

    /** checkout the committed file whose name is NAME
     *  from the commit with sha1 code CODE.*/
    public static void checkoutCommittedFileV2(String code, String name) {
        String id = code.substring(0, 6);
        File gitlet = new File(".gitlet");
        CommitTree commitTree = readObject(gitlet, CommitTree.class);
        for (Commit commit : commitTree.getCommitRecord()) {
            String sha1 = commit.getSha1().substring(0, 6);
            if (sha1.equals(id)) {
                if (!commit.getContent().containsKey(name)) {
                    System.out.println("File does not exist in that commit.");
                } else {
                    Blob blob = commit.getContent().get(name);
                    overwrite(name, blob.getBytes());
                }
                return;
            }
        }
        System.out.println("No commit with that id exists.");
        writeObject(gitlet, commitTree);
    }


    /** checkout the branch whose name is NAME.*/
    public static void checkoutBranchV3(String name) {
        File gitlet = new File(".gitlet");
        CommitTree commitTree = readObject(gitlet, CommitTree.class);
        if (!commitTree.getBranches().containsKey(name)) {
            System.out.println("No such branch exists.");
        } else if (commitTree.getHeadPointer().equals(name)) {
            System.out.println("No need to checkout the current branch.");
        } else {
            List<String> currentFiles = plainFilenamesIn(".");
            List<String> tracked = new ArrayList<>();
            tracked.addAll(commitTree.getStage().keySet());
            tracked.addAll(commitTree.getLastCommit().getContent().keySet());
            Commit required = commitTree.getBranches().get(name);
            for (HashMap.Entry<String, Blob> entry
                    : required.getContent().entrySet()) {
                String bName = entry.getKey();
                Blob blob = entry.getValue();
                if (currentFiles.contains(bName)) {
                    if (!tracked.contains(bName)) {
                        System.out.println("There is an untracked file "
                                + "in the way; delete it or add it first.");
                        return;
                    }
                }
                overwrite(bName, blob.getBytes());
            }

            for (String fName : tracked) {
                if (!required.getContent().containsKey(fName)) {
                    File file = new File(fName);
                    file.delete();
                }
            }

            commitTree.setBranches(commitTree.getHeadPointer(),
                    commitTree.getLastCommit());
            commitTree.setHeadPointer(name);
            commitTree.setLastCommit(required);
            commitTree.clearStage();
        }

        writeObject(gitlet, commitTree);


    }


    /** remove BRANCH from the current commit tree.*/
    public static void rmBranch(String branch) {
        File gitlet = new File(".gitlet");
        CommitTree commitTree = readObject(gitlet, CommitTree.class);
        if (!commitTree.getBranches().containsKey(branch)) {
            System.out.println(" A branch with that name does not exist.");
        } else if (commitTree.getHeadPointer().equals(branch)) {
            System.out.println("Cannot remove the current branch.");
        } else {
            commitTree.getBranches().remove(branch);
        }

        writeObject(gitlet, commitTree);
    }



    /** reset to the commit with RESETID.*/
    public static void reset(String resetID) {
        String id = resetID.substring(0, 6);
        File gitlet = new File(".gitlet");
        CommitTree commitTree = readObject(gitlet, CommitTree.class);
        for (Commit commit : commitTree.getCommitRecord()) {
            if (commit.getSha1().substring(0, 6).equals(id)) {
                List<String> currentFiles = plainFilenamesIn(".");
                List<String> tracked = new ArrayList<>();
                tracked.addAll(commitTree.getStage().keySet());
                tracked.addAll(commitTree.getLastCommit()
                        .getContent().keySet());

                HashMap<String, String> qualified = new HashMap<>();
                for (String name : currentFiles) {
                    if (commit.getContent().keySet().contains(name)) {
                        if (!tracked.contains(name)) {
                            System.out.println("There is an untracked file "
                                    + "in the way; delete it or add it first.");
                            return;
                        }
                    } else {
                        if (tracked.contains(name)
                                && !commit.getContent().containsKey(name)) {
                            qualified.put(name, "delete");
                        }
                    }
                }

                for (HashMap.Entry<String, String> entry
                        : qualified.entrySet()) {
                    String name = entry.getKey();
                    String method = entry.getValue();
                    if (method.equals("delete")) {
                        File file = new File(name);
                        file.delete();
                    }
                }

                for (HashMap.Entry<String, Blob> entry
                        : commit.getContent().entrySet()) {
                    String name = entry.getKey();
                    Blob blob = entry.getValue();
                    overwrite(name, blob.getBytes());
                }


                commitTree.clearStage();
                commitTree.getRemoveFiles().clear();
                commitTree.setLastCommit(commit);
                writeObject(gitlet, commitTree);
                return;
            }
        }
        System.out.println("No commit with that id exists.");

    }

    /** handle the failure cases met when
     *  merging BRANCH2 to the current branch and
     *  return a boolean. */
    private static boolean mergeException(String branch2) {
        File gitlet = new File(".gitlet");
        CommitTree commitTree = readObject(gitlet, CommitTree.class);
        String branch1 = commitTree.getHeadPointer();
        if (commitTree.getStage().size() != 0
                || commitTree.getRemoveFiles().size() != 0) {
            System.out.println("You have uncommitted changes.");
            return true;
        } else if (!commitTree.getBranches().containsKey(branch2)) {
            System.out.println("A branch with that name does not exist.");
            return true;
        } else if (branch2.equals(branch1)) {
            System.out.println("Cannot merge a branch with itself.");
            return true;
        }
        ArrayList<Commit> ancestors1 = commitTree
                .getLastCommit().findAncestors();
        ArrayList<Commit> ancestors2 = commitTree
                .getBranches().get(branch2).findAncestors();
        if (ancestors1.contains(commitTree.getBranches().get(branch2))) {
            System.out.println("Given branch "
                    + "is an ancestor of the current branch.");
            return true;
        } else if (ancestors2.contains(commitTree.getLastCommit())) {
            commitTree.removeBranch(branch1);
            commitTree.setHeadPointer(branch2);
            commitTree.setLastCommit(commitTree.getBranches().get(branch2));
            System.out.println("Current branch fast-forwarded.");
            return true;
        }
        return false;
    }

    /** merge BRANCH2 to the current branch.*/
    public static void merge(String branch2) {
        File gitlet = new File(".gitlet");
        CommitTree commitTree = readObject(gitlet, CommitTree.class);
        String branch1 = commitTree.getHeadPointer();
        if (mergeException(branch2)) {
            return;
        }
        Commit splitPoint = findSplitPoint(branch2);
        Commit commit1 = commitTree.getLastCommit();
        Commit commit2 = commitTree.getBranches().get(branch2);
        List<String> currentFiles = plainFilenamesIn(".");
        List<String> tracked = new ArrayList<>();
        tracked.addAll(commitTree.getStage().keySet());
        tracked.addAll(commitTree.getLastCommit().getContent().keySet());
        List<String> untracked = new ArrayList<>();
        for (String string : currentFiles) {
            if (!tracked.contains(string)) {
                untracked.add(string);
            }
        }

        HashMap<String, String> toBeImplemented = new HashMap<>();
        situationAfterSplit(splitPoint, commit1,
                commit2, untracked, toBeImplemented);
        if (toBeImplemented.containsValue("stop")) {
            return;
        } else {
            for (HashMap.Entry<String, String> entry
                    : toBeImplemented.entrySet()) {
                String fName = entry.getKey();
                String method = entry.getValue();
                if (method.equals("conflict")) {
                    Blob blob1 = null;
                    Blob blob2 = null;
                    if (commit1.getContent().containsKey(fName)) {
                        blob1 = commit1.getContent().get(fName);
                    }
                    if (commit2.getContent().containsKey(fName)) {
                        blob2 = commit2.getContent().get(fName);
                    }
                    Blob blob = dealWithConflict(fName, blob1, blob2);
                    overwrite(fName, blob.getBytes());
                    commitTree.stageAdd(fName, blob);
                } else if (method.equals("remove")) {
                    removeFile(commitTree, fName);
                    File file = new File(fName);
                    file.delete();
                } else if (method.equals("overwrite")) {
                    overwrite(commitTree, fName, commit2);
                }
            }
        }

        writeObject(gitlet, commitTree);
        commit("Merged " + branch2 + " into " + branch1 + ".");
        commitTree = readObject(gitlet, CommitTree.class);
        commitTree.getLastCommit().setSecondParent(commit2);
        writeObject(gitlet, commitTree);

    }

    /** find the split point of BRANCH2 and the head branch
     *  of the current commit tree.
     *  return this commit. */
    public static Commit findSplitPoint(String branch2) {
        File gitlet = new File(".gitlet");
        CommitTree commitTree = readObject(gitlet, CommitTree.class);
        assert !branch2.equals(commitTree.getHeadPointer());
        Commit commit1 = commitTree.getLastCommit();
        Commit commit2 = commitTree.getBranches().get(branch2);
        ArrayList<Commit> ancestors1 = commit1.findAncestors();
        ArrayList<Commit> ancestors2 = commit2.findAncestors();
        for (int i = 0; i < ancestors1.size(); i += 1) {
            Commit commit = ancestors1.get(i);
            if (ancestors2.contains(commit)) {
                return ancestors1.get(i);
            }
        }
        return null;
    }



    /** use IMPLEMENT to record the file we need to deal with
     *  when we want to merge C1 and C2, whose split point
     *  is SPLITPOINT. If a file we want to change is in UNTRACKED
     *  stop here and return. */
    public static void situationAfterSplit(Commit splitPoint, Commit c1, Commit
            c2, List<String> untracked, HashMap<String, String> implement) {
        HashMap<String, Blob> splitContent = splitPoint.getContent();
        HashMap<String, Blob> content1 = c1.getContent();
        HashMap<String, Blob> content2 = c2.getContent();
        for (HashMap.Entry<String, Blob> entry : splitContent.entrySet()) {
            String name = entry.getKey();
            String primitive = entry.getValue().getSha1();
            if (content1.containsKey(name)) {
                String v1 = content1.get(name).getSha1();
                if (content2.containsKey(name)) {
                    String v2 = content2.get(name).getSha1();
                    if (!v2.equals(primitive) && v1.equals(primitive)) {
                        if (untracked.contains(v2)) {
                            System.out.println("There is an untracked file in "
                                    + "the way; delete it or add it first.");
                            implement.put(name, "stop");
                            return;
                        }
                        implement.put(name, "overwrite");
                    } else if (!v1.equals(primitive) && !v2.equals
                            (primitive) && !v1.equals(v2)) {
                        implement.put(name, "conflict");
                    }
                } else {
                    if (v1.equals(primitive)) {
                        implement.put(name, "remove");
                    } else {
                        implement.put(name, "conflict");
                    }
                }
            } else {
                if (content2.containsKey(name)) {
                    String version2 = content2.get(name).getSha1();
                    if (!version2.equals(primitive)) {
                        implement.put(name, "conflict");
                    }
                }
            }
        }
        for (HashMap.Entry<String, Blob> entry : content2.entrySet()) {
            String name = entry.getKey();
            if (!splitContent.containsKey(name)) {
                String version2 = entry.getValue().getSha1();
                if (!content1.containsKey(name)) {
                    if (untracked.contains(name)) {
                        System.out.println("There is an untracked file in "
                                + "the way; delete it or add it first.");
                        implement.put(name, "stop");
                        return;
                    }
                    implement.put(name, "overwrite");
                } else if (content1.containsKey(name)) {
                    String version1 = content1.get(name).getSha1();
                    if (!version1.equals(version2)) {
                        implement.put(name, "conflict");
                    }
                }
            }
        }
    }

    /** Remove the file whose name is NAME from the staging
     *  area of COMMITTREE.*/
    private static void removeFile(CommitTree commitTree, String name) {
        commitTree.stageRemove(name);
        commitTree.setRemoveFiles(name);
    }

    /** Overwrite a file whose name is NAME by COMMIT2 and add it to the
     *  staging area of COMMITTREE.*/
    private static void overwrite(CommitTree commitTree,
                                  String name, Commit commit2) {
        overwrite(name, commit2.getContent().get(name).getBytes());
        commitTree.stageAdd(name, commit2.getContent().get(name));
    }


    /** Merge two blobs BLOB1 and BLOB2 whose common name is NAME.
     *  Return the new blob. */
    private static Blob dealWithConflict(String name, Blob blob1, Blob blob2) {
        System.out.println("Encountered a merge conflict.");
        String content1 = "";
        String content2 = "";
        if (blob1 != null) {
            content1 = new String(blob1.getBytes());
        }
        if (blob2 != null) {
            content2 = new String(blob2.getBytes());
        }
        String result = "<<<<<<< HEAD\n" + content1
                + "=======\n" + content2 + ">>>>>>>\n";
        return new Blob(result.getBytes(), name);
    }


}
