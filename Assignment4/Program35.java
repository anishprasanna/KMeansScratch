import java.util.Scanner;

public class FizzBuzzGame {
    public static void main (String [] args)
    {
        // ask user to input 3 integers between 1 and 100 in increasing order (X Y N)

        Scanner sc = new Scanner(System.in);
        System.out.print("Input three integers between 1 and 100: ");
        int x = sc.nextInt();
        int y = sc.nextInt();
        int n = sc.nextInt(); // this integer will be the final bound of the loop (n)

        // output ever integer from 1 to n in order with appropriate replacements

        int i;                                          // counter variable for loop

        if ((x < 1) || (y < 1) || (n < 1)) {            // will output invalid if the user input doesn't meet parameters
            System.out.println("Invalid entry");
        } else {
            for (i = 1; i <= n; i++) {                  // loops through every integer from 1 to n
                if (i % (x * y) == 0) {                 // integers divisible by both x and y are replaced with FizzBuzz
                    System.out.println("FizzBuzz");
                } else if (i % x == 0) {                // integers divisible by only x are replaced with Fizz
                    System.out.println("Fizz");
                } else if (i % y == 0) {
                    System.out.println("Buzz");         // integers divisible by only y are replaced with Buzz
                } else {
                    System.out.println(i);
                }
            }
        }
    }
}
