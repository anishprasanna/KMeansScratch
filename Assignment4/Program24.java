import java.util.Scanner;

public class FizzBuzzGame {
    public static void main(String[] args) {

        // Creates scanner for user input
        Scanner scnr = new Scanner(System.in);

        // Declares variables
        int divisor1;
        int divisor2;
        int upperBound;

        // Writes welcome message
        System.out.println("Fizz Buzz Game ===========");

        // Writes prompt for user input
        System.out.print("Input three integers: ");

        // Reads the two divisors and upper bound
        divisor1 = scnr.nextInt();
        divisor2 = scnr.nextInt();
        upperBound = scnr.nextInt();

            // Checks each integer to see if it is divisible by either of the two user-given integers
            for (int i = 1; i <= upperBound; i++) {
                if (((i % divisor1) == 0) && ((i % divisor2) == 0)) {
                    // Writes FizzBuzz if divisible by both
                    System.out.print("FizzBuzz");
                } else if ((i % divisor1) == 0) {
                    // Writes Fizz if only divisible by the first number
                    System.out.print("Fizz");
                } else if ((i % divisor2) == 0) {
                    // Writes Buzz if only divisible by the second number
                    System.out.print("Buzz");
                } else {
                    // Writes the integer if not divisible by either
                    System.out.print(i);
                }

                // Starts new line
                System.out.println();
            }
    }
}
