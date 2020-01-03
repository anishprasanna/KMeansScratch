import java.util.Scanner;

public class FizzBuzzGame {
    public static void main (String [] args) {
        Scanner scnr = new Scanner(System.in);
        //set up scanner

        int x;
        int y;
        int n;
        int i;
        //define my variables

        System.out.println("Fizz Buzz Game =============");
        System.out.println(" ");
        System.out.println("Input three integers (1-100): ");

        x = scnr.nextInt();
        y = scnr.nextInt();
        n = scnr.nextInt();
        //define my variables

        for( i = 1; i <= n; ++i)   {
            if (((i % x) == 0) && ((i % y) == 0))
                System.out.println("fizzbuzz");
            else if ((i % x) == 0)
                System.out.println("fizz");
            else if ((i % y) == 0)
                System.out.println("buzz");
            else
                System.out.println(i);

        }
        //outputs fizzbuzz for a number divisible by both x and y. fizz for just x and buzz for just y



    }
}
