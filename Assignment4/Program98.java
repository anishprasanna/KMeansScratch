import java.util.Scanner;
public class ReplaceTester {
    public static void main(String [] args)
    {
        System.out.println("Replace Text");
        System.out.println("");
        Scanner scnr = new Scanner(System.in);
        //Define variables
        String originalString;
        char character1;
        char character2;
        String newString;

        //Ask for user input
        System.out.print("Enter a string that you would like to modify: ");
        originalString = scnr.nextLine();
        System.out.print("Enter a character that you want to replace: ");
        character1 = scnr.next().charAt(0);
        System.out.print("Enter a replacement character: ");
        character2 = scnr.next().charAt(0);

        //Change string
        newString = originalString.replace(character1, character2);

        //Print modified string
        System.out.println(newString);

    }
}
