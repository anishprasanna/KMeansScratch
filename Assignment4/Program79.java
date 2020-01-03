import java.util.Scanner;
public class ReplaceTester {
    public static void main(String[] args)
    {
       Scanner scnr = new Scanner(System.in);

        //get user input
        System.out.println("Replace Text =================");
        System.out.print("Enter a string that you want to modify: ");
        String text = scnr.nextLine();
        System.out.print("Enter a character that you want to replace: ");
        String character1 = scnr.nextLine();
        System.out.print("Enter a replacement character: ");
        String character2 = scnr.nextLine();

       //modify string
        text = text.replace(character1, character2);

        //print new string
        System.out.println("The modified string is: " + text);
    }

}
