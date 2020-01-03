import java.util.Scanner;
public class ReplaceTester {
    public static void main(String [] arg){
        Scanner scnr = new Scanner(System.in);

        String phrase;

        char a;
        char b;

        System.out.println("Enter a string that you want to modify: ");
        phrase = scnr.next();

        System.out.println("Enter a character that you want to replace: ");
        a = scnr.next().charAt(0);

        System.out.println("Enter a replacement character: ");
        b = scnr.next().charAt(0);

        System.out.println("The modified string is: " + phrase.replace(a,b));



    }
}
