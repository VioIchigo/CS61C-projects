package gitlet;

import java.io.File;

/** Driver class for Gitlet, the tiny stupid version-control system.
 *  @author Violet Fu
 */
public class Main extends Command {

    /** Usage: java gitlet. Main ARGS, where ARGS contains
     *  <COMMAND> <OPERAND> .... */
    public static void main(String... args) {
        if (args.length == 0) {
            System.out.println("Please enter a command.");
            return;
        }
        String command = args[0];
        File gitlet = new File(".gitlet");
        if (!gitlet.exists() && !command.equals("init")) {
            System.out.println("Not in an initialized Gitlet directory.");
            return;
        }
        switch (command) {
        case "init":
            initMain(args);
            break;
        case "add":
            addMain(args);
            break;
        case "commit":
            commitMain(args);
            break;
        case "rm":
            rmMain(args);
            break;
        case "log":
            logMain(args);
            break;
        case "global-log":
            globalLogMain(args);
            break;
        case "find":
            findMain(args);
            break;
        case "status":
            statusMain(args);
            break;
        case "branch":
            branchMain(args);
            break;
        case "rm-branch":
            rmBranchMain(args);
            break;
        case "reset":
            resetMain(args);
            break;
        case "merge":
            mergeMain(args);
            break;
        case "checkout":
            checkoutMain(args);
            break;
        default:
            System.out.println("No command with that name exists.");
            break;
        }
    }
    /** Performs init command.
     *  It takes in one or more Strings ARGS*/
    private static void initMain(String... args) {
        if (args.length != 1) {
            System.out.println("Incorrect operands.");
            return;
        }
        Command.init();
    }
    /** Performs add command.
     *  It takes in one or more Strings ARGS*/
    private static void addMain(String... args) {
        if (args.length != 2) {
            System.out.println("Incorrect operands.");
            return;
        }
        Command.add(args[1]);
    }
    /** Performs commit command.
     *  It takes in one or more Strings ARGS*/
    private static void commitMain(String... args) {
        if (args.length != 2) {
            System.out.println("Incorrect operands.");
            return;
        }
        if (args[1].length() == 0) {
            System.out.println("Please enter a commit message.");
            return;
        }
        Command.commit(args[1]);
    }
    /** Performs romove command.
     *  It takes in one or more Strings ARGS*/
    private static void rmMain(String... args) {
        if (args.length != 2) {
            System.out.println("Incorrect operands.");
            return;
        }
        Command.remove(args[1]);
    }
    /** Performs log command.
     *  It takes in one or more Strings ARGS*/
    private static void logMain(String... args) {
        if (args.length != 1) {
            System.out.println("Incorrect operands.");
            return;
        }
        Command.log();
    }
    /** Performs global-log command.
     *  It takes in one or more Strings ARGS*/
    private static void globalLogMain(String... args) {
        if (args.length != 1) {
            System.out.println("Incorrect operands.");
            return;
        }
        Command.globalLog();
    }
    /** Performs find command.
     *  It takes in one or more Strings ARGS*/
    private static void findMain(String... args) {
        if (args.length != 2) {
            System.out.println("Incorrect operands.");
            return;
        }
        Command.find(args[1]);
    }
    /** Performs status command.
     *  It takes in one or more Strings ARGS*/
    private static void statusMain(String... args) {
        if (args.length != 1) {
            System.out.println("Incorrect operands.");
            return;
        }
        Command.status();
    }
    /** Performs branch command.
     *  It takes in one or more Strings ARGS*/
    private static void branchMain(String... args) {
        if (args.length != 2) {
            System.out.println("Incorrect operands.");
            return;
        }
        Command.branch(args[1]);
    }
    /** Performs rm-branch command.
     *  It takes in one or more Strings ARGS*/
    private static void rmBranchMain(String... args) {
        if (args.length != 2) {
            System.out.println("Incorrect operands.");
            return;
        }
        Command.rmBranch(args[1]);
    }
    /** Performs reset command.
     *  It takes in one or more Strings ARGS*/
    private static void resetMain(String... args) {
        if (args.length != 2) {
            System.out.println("Incorrect operands.");
            return;
        }
        Command.reset(args[1]);
    }
    /** Performs merge command.
     *  It takes in one or more Strings ARGS*/
    private static void mergeMain(String... args) {
        if (args.length != 2) {
            System.out.println("Incorrect operands.");
            return;
        }
        Command.merge(args[1]);
    }
    /** Performs checkout command.
     *  It takes in one or more strings ARGS. */
    private static void checkoutMain(String... args) {
        if (args.length > 4) {
            System.out.println("Incorrect operands.");
            return;
        }
        if (args.length == 3) {
            if (!args[1].equals("--")) {
                System.out.println("Incorrect operands.");
                return;
            }
            checkoutFileV1(args[2]);
        } else if (args.length == 4) {
            if (!args[2].equals("--")) {
                System.out.println("Incorrect operands.");
                return;
            }
            checkoutCommittedFileV2(args[1], args[3]);
        } else if (args.length == 2) {
            checkoutBranchV3(args[1]);
        }
    }
}
