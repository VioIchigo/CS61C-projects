package db61b;

import org.junit.Test;
import static org.junit.Assert.*;
import java.util.ArrayList;
import ucb.junit.textui;


/** The suite of all JUnit tests for the qirkat package.
 *  @author P. N. Hilfinger
 */
public class UnitTest {

    @Test
    public void testColumns() {
        Table a = new Table(new String[]{"SID", "Fname", "Lname", "SemEnter"});
        assertEquals(4, a.columns());
    }

    @Test
    public void testAdd() {
        Table a = new Table(new String[] {"SID", "Fname", "Lname", "SemEnter"});
        String[] newRow = new String[] {"101", "Jason", "Knowles", "F"};
        String[] nextRow = new String[] {"102", "Valerie", "Chan", "S"};
        assertEquals(true, a.add(newRow));
        assertEquals(true, a.add(nextRow));
        assertEquals(false, a.add(nextRow));
        assertEquals(2, a.size());
    }

    @Test
    public void testSize() {
        Table a = new Table(new String[] {"SID", "Fname", "Lname", "SemEnter"});
        String[] newRow = new String[] {"101", "Jason", "Knowles", "F"};
        assertEquals(0, a.size());
        a.add(newRow);
        assertEquals(1, a.size());
    }


    @Test
    public void testGet() {
        Table a = new Table(new String[] {"SID", "Fname", "Lname", "SemEnter"});
        String[] newRow = new String[] {"101", "Jason", "Knowles", "F"};
        String[] nextRow = new String[] {"102", "Valerie", "Chan", "S"};
        a.add(newRow);
        a.add(nextRow);
        assertEquals("Valerie", a.get(1, 1));
    }

    @Test
    public void testGetTitle() {
        Table a = new Table(new String[] {"SID", "Fname", "Lname", "SemEnter"});
        String[] newRow = new String[] {"101", "Jason", "Knowles", "F"};
        String[] nextRow = new String[] {"102", "Valerie", "Chan", "S"};
        a.add(newRow);
        a.add(nextRow);
        assertEquals("Lname", a.getTitle(2));
    }

    @Test
    public void testFindColumn() {
        Table a = new Table(new String[] {"SID", "Fname", "Lname", "SemEnter"});
        assertEquals(-1, a.findColumn("First"));
        assertEquals(1, a.findColumn("Fname"));
    }

    @Test
    public void testDataBase() {
        Database a = new Database();
        Table t1 = new Table(new String[] {"Fname", "Lname", "Grades"});
        String[] newRow = new String[] {"Jason", "Knowles", "B"};
        String[] nextRow = new String[] {"Valerie", "Chan", "B+"};
        String[] thirdRow = new String[] {"Yangfan", "Chan", "B"};
        String[] fourthRow = new String[] {"Shana", "Brown", "B+"};
        t1.add(newRow);
        t1.add(nextRow);
        t1.add(thirdRow);
        t1.add(fourthRow);
        a.put("Information1", t1);
        Table getInformation1 = a.get("Information1");
        getInformation1.print();
        Table t2 = new Table(new String[] {"YearEnter", "Major", "SemEnter"});
        String[] row1 = new String[] {"2003", "EECS", "F"};
        String[] row2 = new String[] {"2003", "Math", "S"};
        String[] row3 = new String[] {"2003", "LSUnd", "F"};
        String[] row4 = new String[] {"2004", "EECS", "S"};
        t2.add(row1);
        t2.add(row2);
        t2.add(row3);
        t2.add(row4);
        a.put("Information2", t2);
        Table getInformation2 = a.get("Information2");
        getInformation2.print();
    }

    @Test
    public void testPrint() {
        Table a = new Table(new String[] {"Fname", "Lname", "Grades"});
        String[] newRow = new String[] {"Jason", "Knowles", "B"};
        String[] nextRow = new String[] {"Valerie", "Chan", "B+"};
        String[] thirdRow = new String[] {"Yangfan", "Chan", "B"};
        String[] fourthRow = new String[] {"Shana", "Brown", "B+"};
        a.add(newRow);
        a.add(nextRow);
        a.add(thirdRow);
        a.add(fourthRow);
        a.print();
    }

    @Test
    public void testSecondAdd() {
        Table t1 = new Table(new String[] {"Fname", "Lname", "Grades"});
        String[] newRow = new String[] {"Jason", "Knowles", "B"};
        String[] nextRow = new String[] {"Valerie", "Chan", "B+"};
        String[] thirdRow = new String[] {"Yangfan", "Chan", "B"};
        String[] fourthRow = new String[] {"Shana", "Brown", "B+"};
        t1.add(newRow);
        t1.add(nextRow);
        t1.add(thirdRow);
        t1.add(fourthRow);
        Table t2 = new Table(new String[] {"YearEnter", "Major", "SemEnter"});
        String[] row1 = new String[] {"2003", "EECS", "F"};
        String[] row2 = new String[] {"2003", "Math", "S"};
        String[] row3 = new String[] {"2003", "LSUnd", "F"};
        String[] row4 = new String[] {"2004", "EECS", "S"};
        t2.add(row1);
        t2.add(row2);
        t2.add(row3);
        t2.add(row4);
        String[] columnNames =
                new String[] {"Fname", "Lname", "Major", "SemEnter"};
        ArrayList<Column> collection = new ArrayList<>();
        for (String col : columnNames) {
            Column toAdd = new Column(col, t1, t2);
            collection.add(toAdd);
        }
        Table result = new Table(columnNames);
        for (int i = 0; i < t1.size(); i += 1) {
            for (int j = 0; j < t2.size(); j += 1) {
                result.add(collection, i, j);
            }
        }
        result.print();
        assertEquals(16, result.size());
        assertEquals(4, result.columns());
    }


    @Test
    public void testWriteReadTable() {
        String[] row = new String[] {"Jason", "Knowles", "B"};
        Table t = new Table(new String[]{"SID", "FName", "LName"});
        t.add(row);
        t.writeTable("Information");
        Table a = Table.readTable("Information");
        assertEquals(1, a.size());
    }


    @Test
    public void testSelectOneTable() {
        String[] info;
        info = new String[] {"SID", "Fname", "Lname", "SemEnter", "Grades"};
        Table one = new Table(info);

        String[] newRow = new String[] {"101", "Jason", "Knowles", "F", "B"};
        String[] nextRow = new String[] {"102", "Valerie", "Chan", "S", "B+"};
        String[] thridRow = new String[]{"103", "John", "Xavier", "F", "B+"};
        one.add(nextRow);
        one.add(newRow);
        one.add(thridRow);

        ArrayList<String> columns = new ArrayList<>();
        columns.add("SID");
        columns.add("Grades");

        ArrayList<Condition> condList = new ArrayList<>();
        Column col = new Column("Grades", one);
        Column col2 = new Column("SemEnter", one);
        Condition condition = new Condition(col, ">", "B");
        Condition condition2 = new Condition(col2, "=", "S");
        condList.add(condition);
        condList.add(condition2);

        Table result = one.select(columns, condList);
        result.print();
    }



    @Test
    public void testSelectTwoTable() {
        String[] info;
        info = new String[] {"SID", "Fname", "Lname", "Grades"};
        Table one = new Table(info);

        String[] info2;
        info2 = new String[] {"Lname", "SemEnter", "Grades"};
        Table two = new Table(info2);

        String[] newRow = new String[] {"101", "Jason", "Knowles", "B"};
        String[] nextRow = new String[] {"102", "Valerie", "Chan", "B+"};
        one.add(nextRow);
        one.add(newRow);

        String[] newRow2 = new String[] {"Knowles", "F", "B"};
        String[] nextRow2 = new String[] {"Chan", "S", "B+"};
        two.add(newRow2);
        two.add(nextRow2);

        ArrayList<String> selectFrom = new ArrayList<>();
        selectFrom.add("SemEnter");
        selectFrom.add("Grades");
        selectFrom.add("SID");

        ArrayList<Condition> condList = new ArrayList<>();
        Column col = new Column("Grades", one, two);
        Column col2 = new Column("SID", one, two);
        Condition condition = new Condition(col, ">=", "B");
        Condition condition2 = new Condition(col2, "<", "104");
        condList.add(condition);
        condList.add(condition2);

        Table result = one.select(two, selectFrom, condList);
        result.print();
    }

    @Test
    public void testDifferent() {
        Database a = new Database();
        Table t1 = new Table(new String[]{"Fname", "Lname", "Grades"});
        String[] newRow = new String[]{"Jason", "Knowles", "B"};
        String[] nextRow = new String[]{"Valerie", "Chan", "B+"};
        String[] thirdRow = new String[]{"Yangfan", "Chan", "B"};
        String[] fourthRow = new String[]{"Shana", "Brown", "B+"};
        t1.add(newRow);
        t1.add(nextRow);
        t1.add(thirdRow);
        t1.add(fourthRow);
        Table t2 = new Table(new String[]{"YearEnter", "Major", "SemEnter"});
        String[] row1 = new String[]{"2003", "EECS", "F"};
        String[] row2 = new String[]{"2003", "Math", "S"};
        String[] row3 = new String[]{"2003", "LSUnd", "F"};
        String[] row4 = new String[]{"2004", "EECS", "S"};
        t2.add(row1);
        t2.add(row2);
        t2.add(row3);
        t2.add(row4);
        ArrayList<String> sel = new ArrayList<>();
        sel.add("Fname");
        sel.add("Lname");
        sel.add("Grades");
        sel.add("YearEnter");
        sel.add("Major");
        Table result = t2.select(t1, sel, null);
        result.print();
    }

    /** Run the JUnit tests in this package. Add xxxTest.class entries to
     *  the arguments of runClasses to run other JUnit tests. */
    public static void main(String[] ignored) {
        /* textui.runClasses(); */
        System.exit(textui.runClasses(UnitTest.class));
    }

}
