import java.util.Scanner;
public class OrderedIntegers {
    public static void main(String [] args)
    {
        Scanner scnr = new Scanner(System.in);
        System.out.println("Order Three Integers");
        System.out.println("");

        //Define variables
        int a;
        int b;
        int c;

        //Ask for user input
        System.out.print("Enter three integers: ");
        a = scnr.nextInt();
        b = scnr.nextInt();
        c = scnr.nextInt();

        //Sort integers
        if ( a >= b ) {
            if ( b >= c ) {
                System.out.print("The integers in decreasing order are: " + a + " " + b + " " + c);
            }
            else if ( c >= a ) {
                System.out.print("The integers in decreasing order are: " + c + " " + a + " " + b);
            }
            else {
                System.out.print("The integers in decreasing order are: " + a + " " + c + " " + b);
            }
        }
        else if ( b >= c ) {
            if ( c >= a ) {
                System.out.print("The integers in decreasing order are: " + b + " " + c + " " + a);
            }
            else if ( a >= b ) {
                System.out.print("The integers in decreasing order are: " + a + " " + b + " " + c);
            }
            else {
                System.out.print("The integers in decreasing order are: " + b + " " + a + " " + c);
            }
        }
        else {
            if ( c >= b ) {
                System.out.print("The integers in decreasing order are: " + a + " " + c + " " + b);
            }
            else if ( b >= a ) {
                System.out.print("The integers in decreasing order are: " + b + " " + a + " " + c);
            }
            else {
                System.out.print("The integers in decreasing order are: " + a + " " + b + " " + c);
            }
        }

        //Print integers in decreasing order
    }
}
