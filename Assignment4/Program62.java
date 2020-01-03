import java.util.Scanner;

public class CountInputLength {
    public static void main(String[] args) {
        Scanner scnr = new Scanner(System.in);
        System.out.println("Count Input Length =============");
        //set the variable enterString and let the user to initialize it.
        System.out.print("Enter a line of text: ");
        String enterString = scnr.nextLine();
        // replace space, periods and commas, meaning to delete them from the string
        String string1 = enterString.replace(" ", "");
        String string2 = enterString.replace(".", "");
        String string3 = enterString.replace(",", "");
        //calculate the length fo the string
        int length = string3.length();

        System.out.println("The text length without spaces, periods, or commas is: " + length);


    }
}
*/
import java.util.Scanner;
public class CountInputLength {
    public static void main(String[] args) {
        Scanner scnr = new Scanner(System.in);
        System.out.println("Count Input Length =============");
        //set the variable enterString and let the user to initialize it.
        System.out.print("Enter a line of text: ");
        String enterString = scnr.nextLine();
        int i;
       int lengthNum = 0;
       //using for loop to check and calculate whether the character is , . or space
        for (i=0; i < enterString.length(); i++){
            Character character1  = enterString.charAt(i);
        if (character1 != ' ' && character1!= ',' && character1 != '.'){
            // if the character is not a space, a , or an.  add 1 to the length number.
            lengthNum += 1;

        }

        }
        //type out the output
        System.out.println("The text length without spaces, periods, or commas is: " + lengthNum);
        }
    }
