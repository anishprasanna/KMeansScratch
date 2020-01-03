import java.util.Scanner;
public class CountInputLength {
    public static void main (String[] arg){
        //create variables and scanner
        String inputString;
        int stringLength;
        Scanner scnr = new Scanner(System.in);
        System.out.println( "Enter a line of text: " );
        inputString = scnr.nextLine();
        //remove all spaces, commas, and periods
        inputString = inputString.replace(" ","");
        inputString = inputString.replace(",","");
        inputString = inputString.replace(".","");
        //assign the length to stringLength
        stringLength = inputString.length();
        //Print the result
        System.out.println( "The text length without spaces, periods, or commas is: " + stringLength);
    }
}
