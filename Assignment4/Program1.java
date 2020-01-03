import java.util.Scanner;
public class Program1 {
    public static void main(String[] args) {
        Scanner scnr = new Scanner(System.in);
        System.out.println("Count Input Length ===========");
        System.out.print("Print a line of text: ");

        //get string
        String s = scnr.nextLine();
        int stringLength = 0;

        //find char at i
        for (int i = 0; i < s.length(); i++) {
            //if char at i != " ", ",", "." or "!" add one to length count
            if ((s.charAt(i) != ' ') && (s.charAt(i) != ',') && (s.charAt(i) != '.') && (s.charAt(i) != '!')) {
                stringLength++;
            }
        }

        System.out.println("The text length without spaces, periods, or commas is: " + stringLength);
    }
}
