import java.util.Scanner;

public class FizzBuzzGame {
    public static void main(String[] args) {

        Scanner scnr = new Scanner(System.in);
        int X; // Defines one of three integers entered
        int Y; // Defines one of three integers entered
        int N; // Defines one of three integers entered

        System.out.print("Enter three integers from least to greatest: "); // Prompts user to enter integers from least to greatest
        X = scnr.nextInt();
        Y = scnr.nextInt();
        N = scnr.nextInt();


        for (int i = 1; i <= 100; i++) { // creates a loop that iterates until it reaches 100
            if (i % X == 0) {
                System.out.println("Fizz"); // Outputs Fizz if divisible by X
            } else if (i % Y == 0) {
                System.out.println("Buzz"); // Outputs Buzz if divisible by Y
            } else if (i % N == 0) {
                System.out.println("FizzBuzz"); // Outputs FizzBuzz if divisible by both X and Y
            } else {
                System.out.println(i); // Outputs an integer if not divisible by X or Y


            }


        }
    }
}
