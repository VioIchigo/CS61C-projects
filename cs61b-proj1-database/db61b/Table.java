package db61b;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

import static db61b.Utils.*;

/** A single table in a database.
 *  @author P. N. Hilfinger
 */
class Table {
    /** A new Table whose columns are given by COLUMNTITLES, which may
     *  not contain duplicate names. */
    Table(String[] columnTitles) {
        if (columnTitles.length == 0) {
            throw error("table must have at least one column");
        }
        _size = 0;
        _rowSize = columnTitles.length;

        for (int i = columnTitles.length - 1; i >= 1; i -= 1) {
            for (int j = i - 1; j >= 0; j -= 1) {
                if (columnTitles[i].equals(columnTitles[j])) {
                    throw error("duplicate column name: %s",
                            columnTitles[i]);
                }
            }
        }

        _titles = columnTitles;
        _columns = new ValueList[_rowSize];
        for (int i = 0; i < _rowSize; i += 1) {
            _columns[i] = new ValueList();
        }

    }

    /** A new Table whose columns are give by COLUMNTITLES. */
    Table(List<String> columnTitles) {
        this(columnTitles.toArray(new String[columnTitles.size()]));
    }

    /** Return the number of columns in this table. */
    public int columns() {
        return _titles.length;
    }

    /** Return the title of the Kth column.  Requires 0 <= K < columns(). */
    public String getTitle(int k) {
        return _titles[k];
    }

    /** Return the number of the column whose title is TITLE, or -1 if
     *  there isn't one. */
    public int findColumn(String title) {
        for (int i = 0; i < _rowSize; i += 1) {
            if (title.equals(getTitle(i))) {
                return i;
            }
        }
        return -1;
    }

    /** Return the number of rows in this table. */
    public int size() {
        return _size;
    }


    /** Return the value of column number COL (0 <= COL < columns())
     *  of record number ROW (0 <= ROW < size()). */
    public String get(int row, int col) {
        try {
            return _columns[col].get(row);
        } catch (IndexOutOfBoundsException excp) {
            throw error("invalid row or column");
        }
    }
    /** Add a new row whose column values are VALUES to me if no equal
     *  row already exists.  Return true if anything was added,
     *  false otherwise. */
    public boolean add(String[] values) {
        boolean added = false;
        if (values.length != _rowSize) {
            throw error("there must be exactly one "
                    + "literal for each column of the table");
        } else if (existed(values)) {
            return false;
        } else {
            _size += 1;
            for (int i = 0; i < _rowSize; i += 1) {
                _columns[i].add(values[i]);
                added = true;
            }

            int ind = _size - 1;
            for (int row = 0; row < _size - 1; row += 1) {
                if (compareRows(_size - 1, row) < 0) {
                    int record = _index.get(row);
                    _index.remove(row);
                    _index.add(row, record + 1);
                    ind -= 1;
                }
            }
            _index.add(ind);
        }
        return added;
    }

    /** Determine whether a new String[] ROW is in the table.
     *  The method returns true if the ROW has already existed
     *  else return false. */
    private boolean existed(String[] row) {
        if (_size == 0) {
            return false;
        } else {
            for (int j = 0; j < _size; j += 1) {
                ArrayList<String> record = new ArrayList<>();
                for (int i = 0; i < _rowSize; i += 1) {
                    if (row[i].equals(get(j, i))) {
                        record.add(row[i]);
                    }
                }
                if (record.size() == _rowSize) {
                    return true;
                }
            }
        }
        return false;
    }

    /** Add a new row whose column values are extracted by COLUMNS from
     *  the rows indexed by ROWS, if no equal row already exists.
     *  Return true if anything was added, false otherwise. See
     *  Column.getFrom(Integer...) for a description of how Columns
     *  extract values. */
    public boolean add(List<Column> columns, Integer... rows) {
        String[] row = new String[_rowSize];
        int i = 0;
        for (Column column: columns) {
            row[i] = column.getFrom(rows);
            i += 1;
        }
        return add(row);
    }

    /** Read the contents of the file NAME.db, and return as a Table.
     *  Format errors in the .db file cause a DBException. */
    static Table readTable(String name) {
        BufferedReader input;
        Table table;
        input = null;
        table = null;
        try {
            input = new BufferedReader(new FileReader(name + ".db"));
            String header = input.readLine();
            if (header == null) {
                throw error("missing header in DB file");
            }
            String[] columnNames = header.split(",");
            table = new Table(columnNames);
            String nextRow = input.readLine();
            while (nextRow != null) {
                String[] eachRow = nextRow.split(",");
                table.add(eachRow);
                nextRow = input.readLine();
            }
        } catch (FileNotFoundException e) {
            throw error("could not find %s.db", name);
        } catch (IOException e) {
            throw error("problem reading from %s.db", name);
        } finally {
            if (input != null) {
                try {
                    input.close();
                } catch (IOException e) {
                    /* Ignore IOException */
                }
            }
        }
        return table;
    }

    /** Write the contents of TABLE into the file NAME.db. Any I/O errors
     *  cause a DBException. */
    void writeTable(String name) {
        PrintStream output;
        output = null;
        try {
            String sep;
            sep = "";
            output = new PrintStream(name + ".db");

            for (int i = 0; i < _rowSize - 1; i += 1) {
                sep = sep + _titles[i] + ",";
            }
            sep = sep + _titles[_rowSize - 1];
            output.println(sep);
            for (int row = 0; row < _size; row += 1) {
                output.println(writeRows(row));
            }
        } catch (IOException e) {
            throw error("trouble writing to %s.db", name);
        } finally {
            if (output != null) {
                output.close();
            }
        }
    }

    /** a method that takes in an integer INDEX and returns the
     *  string representation of a row. */
    private String writeRows(int index) {
        String result = "";
        for (int i = 0; i < _rowSize - 1; i += 1) {
            result = result + get(index, i) + ",";
        }
        result = result + get(index, _rowSize - 1);
        return result;
    }

    /** Print my contents on the standard output, separated by spaces
     *  and indented by two spaces. */
    void print() {
        if (_size != 0) {
            for (int i = 0; i < _size; i += 1) {
                String record = "  " + get(_index.indexOf(i), 0);
                for (int j = 1; j < _rowSize; j += 1) {
                    record = record + " " + get(_index.indexOf(i), j);
                }
                System.out.println(record);
            }
        }
    }


    /** Return a new Table whose columns are COLUMNNAMES, selected from
     *  rows of this table that satisfy CONDITIONS. */
    Table select(List<String> columnNames, List<Condition> conditions) {
        for (String name : columnNames) {
            if (findColumn(name) < 0) {
                throw error("Invalid column name");
            }
        }
        Table result = new Table(columnNames);
        ArrayList<Integer> index = new ArrayList<>();
        for (String name : columnNames) {
            index.add(findColumn(name));
        }

        ArrayList<Integer> record = new ArrayList<>();
        for (int row = 0; row < _size; row += 1) {
            if (conditions == null) {
                record.add(row);
            } else {
                if (Condition.test(conditions, row)) {
                    record.add(row);
                }
            }
        }
        for (int i: record) {
            int k = 0;
            String[] rowValue = new String[columnNames.size()];
            for (int j: index) {
                rowValue[k] = get(i, j);
                k += 1;
            }
            result.add(rowValue);
        }

        return result;
    }

    /** Return a new Table whose columns are COLUMNNAMES, selected
     *  from pairs of rows from this table and from TABLE2 that match
     *  on all columns with identical names and satisfy CONDITIONS. */
    Table select(Table table2, List<String> columnNames,
                 List<Condition> conditions) {

        Table result = new Table(columnNames);
        ArrayList<String> common = new ArrayList<>();
        for (String name: _titles) {
            int a = table2.findColumn(name);
            if (a >= 0) {
                common.add(name);
            }
        }
        ArrayList<Column> collection = new ArrayList<>();
        for (String col : columnNames) {
            Column toAdd = new Column(col, this, table2);
            collection.add(toAdd);
        }

        ArrayList<Column> one = new ArrayList<>();
        ArrayList<Column> two = new ArrayList<>();
        for (String name : common) {
            one.add(new Column(name, this));
            two.add(new Column(name, table2));
        }

        for (int i = 0; i < _size; i += 1) {
            for (int j = 0; j < table2.size(); j += 1) {
                if (equijoin(one, two, i, j)) {
                    if (conditions == null) {
                        result.add(collection, i, j);
                    } else {
                        if (Condition.test(conditions, i, j)) {
                            result.add(collection, i, j);
                        }
                    }
                }
            }
        }

        return result;

    }

    /** Return <0, 0, or >0 depending on whether the row formed from
     *  the elements _columns[0].get(K0), _columns[1].get(K0), ...
     *  is less than, equal to, or greater than that formed from elememts
     *  _columns[0].get(K1), _columns[1].get(K1), ....  This method ignores
     *  the _index. */
    private int compareRows(int k0, int k1) {
        for (int i = 0; i < _columns.length; i += 1) {
            int c = _columns[i].get(k0).compareTo(_columns[i].get(k1));
            if (c != 0) {
                return c;
            }
        }
        return 0;
    }

    /** Return true if the columns COMMON1 from ROW1 and COMMON2 from
     *  ROW2 all have identical values.  Assumes that COMMON1 and
     *  COMMON2 have the same number of elements and the same names,
     *  that the columns in COMMON1 apply to this table, those in
     *  COMMON2 to another, and that ROW1 and ROW2 are indices, respectively,
     *  into those tables. */
    private static boolean equijoin(List<Column> common1, List<Column> common2,
                                    int row1, int row2) {
        if (common1.size() != common2.size()) {
            throw error("COMMON1 and COMMON2 must have the "
                    + "same number of elements");
        }

        for (int i = 0; i < common1.size(); i += 1) {
            String record = common2.get(i).getFrom(row2);
            if (!common1.get(i).getFrom(row1).equals(record)) {
                return false;
            }
        }
        return true;

    }

    /** A class that is essentially ArrayList<String>.  For technical reasons,
     *  we need to encapsulate ArrayList<String> like this because the
     *  underlying design of Java does not properly distinguish between
     *  different kinds of ArrayList at runtime (e.g., if you have a
     *  variable of type Object that was created from an ArrayList, there is
     *  no way to determine in general whether it is an ArrayList<String>,
     *  ArrayList<Integer>, or ArrayList<Object>).  This leads to annoying
     *  compiler warnings.  The trick of defining a new type avoids this
     *  issue. */
    private static class ValueList extends ArrayList<String> {
    }

    /** My column titles. */
    private final String[] _titles;
    /** My columns. Row i consists of _columns[k].get(i) for all k. */
    private final ValueList[] _columns;

    /** Rows in the database are supposed to be sorted. To do so, we
     *  have a list whose kth element is the index in each column
     *  of the value of that column for the kth row in lexicographic order.
     *  That is, the first row (smallest in lexicographic order)
     *  is at position _index.get(0) in _columns[0], _columns[1], ...
     *  and the kth row in lexicographic order in at position _index.get(k).
     *  When a new row is inserted, insert its index at the appropriate
     *  place in this list.
     *  (Alternatively, we could simply keep each column in the proper order
     *  so that we would not need _index.  But that would mean that inserting
     *  a new row would require rearranging _rowSize lists (each list in
     *  _columns) rather than just one. */
    private final ArrayList<Integer> _index = new ArrayList<>();

    /** My number of rows (redundant, but convenient). */
    private int _size;
    /** My number of columns (redundant, but convenient). */
    private final int _rowSize;
}
