import java.util.Scanner;
public class CountInputLength {
    public static void main(String[] arg) {
        Scanner scnr = new Scanner(System.in);

        String text;
        char c = ' ';
        int i =0;
        int count=0;

        System.out.println("Enter a line of text: ");
        text = scnr.nextLine();

        for (i=0; i<text.length(); i++) {
            c = text.charAt(i);
            if (c != '.' && c != ',' && c != ' ') {
                count = count + 1;
            }
        }
        System.out.println("The text length without spaces, periods, or commas is: "  + count);


    }
}
