package gitlet;

import java.io.File;
import java.io.Serializable;


/** Object Blob: the contents of a bytes.
 *  @author Violet Fu */

public class Blob extends Utils implements Serializable {

    /** Constructor.*/
    public Blob() {
    }
    /** name of the blob, i.e. name of the file.*/
    private String name;
    /** the content of the file, translated to a list of bytes.*/
    private byte[] bytes;
    /** sha1 code of the file, used to determine whether two files
     *  or two blobs with the same name are identical or not.*/
    private String sha1;

    /** Constructor, takes in a File FILE and returns northing.*/
    public Blob(File file) {
        this.bytes = readContents(file);
        this.name = file.getName();
        sha1 = sha1(bytes);
    }

    /** Constructor, takes in a byte array BYTEARRAY and a String FILENAME
     *  and returns nothing.*/
    public Blob(byte[] byteArray, String fileName) {
        this.bytes = byteArray;
        this.name = fileName;
        sha1 = sha1(bytes);
    }
    /** returns the sha1 code of the blob, reachable from other files.*/
    public String getSha1() {
        return sha1;
    }
    /** returns the byte array of the file.*/
    public byte[] getBytes() {
        return bytes;
    }

    /** returns the name of the blob.*/
    public String getName() {
        return name;
    }
}
