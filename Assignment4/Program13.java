import java.util.Scanner;
public class CountInputLength {
    public static void main (String [] arg)
    {
        Scanner scnr = new Scanner (System.in);

        // identify variables
        String phrase;
        int i;
        int length;
        int newLength = 0;

        // input data
        System.out.print("Enter a line of text: ");
        phrase = scnr.nextLine();
        length = phrase.length();

        //count characters

        for (i = 0; i < length; i++) {
            if ((phrase.charAt(i) != ' ' ) && (phrase.charAt(i) != '.') && (phrase.charAt(i) != ',')) {
               newLength += 1;
            }
        }
        System.out.println("The text length without spaces, periods, or commas is: " + newLength);
    }
}
