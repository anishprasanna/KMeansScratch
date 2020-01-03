import java.util.Scanner;
public class ReplaceTester {
    public static void main (String [] arg)
    {
        Scanner scnr = new Scanner (System.in);
        // identify variables
        String toBeModified;
        char character1;
        char character2;

        //prompt user for data
        System.out.print("Enter a string that you want to modify: ");
        toBeModified = scnr.nextLine();
        System.out.print("Enter a character that you want to replace: ");
        character1 = scnr.next().charAt(0);
        System.out.print("Enter a replacement character: ");
        character2 = scnr.next().charAt(0);

        //replace chosen character
        toBeModified = toBeModified.replace(character1, character2);

        //printResults
        System.out.println("The modified string is: " + toBeModified);



    }

}
