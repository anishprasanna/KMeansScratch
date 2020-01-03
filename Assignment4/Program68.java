import java.util.Scanner;

public class FizzBuzzGame {
    public static void main(String[] args) {
        // Output program title
        System.out.println("Fizz Buzz Game =============\n");

        // Declare variables
        int x = 0;
        int y = 0;
        int n = 0;

        // Read user input
        Scanner sc = new Scanner(System.in);
        System.out.print("Input three integers: ");
        x = sc.nextInt();
        y = sc.nextInt();
        n = sc.nextInt();

        // Ensure that the input meets the condition 1 ≤ X < Y ≤ N ≤ 100
        while ( ! ( (x >= 1) && (y > x) && (n >= y) && (100 >= n) ) ) {
            System.out.println("Please input three integers X, Y, and N, such that 1 ≤ X < Y ≤ N ≤ 100.");
            System.out.print("Input three integers: ");
            x = sc.nextInt();
            y = sc.nextInt();
            n = sc.nextInt();
        }

        // Print the correct output of the integer, Fizz, Buzz, or FizzBuzz for integers from 1 to n
        for (int i = 1; i <= n; i++) {
            if ( (i % x == 0) && (i % y == 0)) {
                System.out.println("FizzBuzz");
            }
            else if (i % x == 0) {
                System.out.println("Fizz");
            }
            else if (i % y == 0) {
                System.out.println("Buzz");
            }
            else {
                System.out.println(i);
            }
        }
    }
}
