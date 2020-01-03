import java.util.Scanner;
public class FizzBuzzGame {
    public static void main(String [] arg) {
        Scanner scnr = new Scanner(System.in);
        System.out.println("Fizz Buzz Game =============");
        System.out.println("");

        //Define variables
        int x;
        int y;
        int n;
        int output = 1;
        System.out.print("Input three integers: ");
        x = scnr.nextInt();
        y = scnr.nextInt();
        n = scnr.nextInt();

        //Compute modulos and output according to FizzBuzz rules
        if ( (1 <= x) && (x < y) && (y <= n) && (n <= 100) ) {
            while ( output <= n ) {
                if ( output % x == 0 ) {
                    if (output % y == 0 ) {
                        System.out.println("FizzBuzz");
                    }
                    else {
                        System.out.println("Fizz");
                    }
                }
                else if (output % y == 0 ) {
                    System.out.println("Buzz");
                }
                else {
                    System.out.println(output);
                }
                output += 1;
            }
        }
    }
}
