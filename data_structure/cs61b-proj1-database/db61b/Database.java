package db61b;

import java.util.HashMap;

/** A collection of Tables, indexed by name.
 *  @author Violet Fu */
class Database {
    /** An empty database. */
    public Database() {
        _table = new HashMap<String, Table>();
    }

    /** Return the Table whose name is NAME stored in this database, or null
     *  if there is no such table. */
    public Table get(String name) {
        return _table.get(name);
    }

    /** Set or replace the table named NAME in THIS to TABLE.  TABLE and
     *  NAME must not be null, and NAME must be a valid name for a table. */
    public void put(String name, Table table) {
        if (name == null || table == null) {
            throw new IllegalArgumentException("null argument");
        } else {
            if (validName(name)) {
                _table.put(name, table);
            } else {
                return;
            }
        }
    }

    /** Test whether a name is valid or not.
     *  a valid NAME is a sequence of letters, digits,
     *  and underscores that does not start with a digit.
     *  The method returns true if it is valid, or false
     *  otherwise. */
    private boolean validName(String name) {
        if (Character.isDigit(name.charAt(0))) {
            return false;
        }
        return true;
    }

    /** The record _table that is convenient to use. */
    private HashMap<String, Table> _table;
}
