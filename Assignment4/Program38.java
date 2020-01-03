import java.util.Scanner;

public class FizzBuzzGame {
    public static void main (String [] args) {
        int xFizzint;
        int yBuzzint;
        int nLimit;
        int numCounter = 1;
        boolean gameStillGoing = true;
        //This is the variables I used in the program. This includes the first fizz int, the second buzz int, the nlimit int, the numCounter int, and the boolean that indicates whether the game is still going

        Scanner scnr = new Scanner (System.in);
        System.out.println("Fizz Buzz Game ====================");
        System.out.println(" ");
        System.out.print("Input three integers: ");
        xFizzint = scnr.nextInt();
        yBuzzint = scnr.nextInt();
        nLimit = scnr.nextInt();
        //This block og code prints out the prompt for the user and takes their inputs

        while (gameStillGoing) {
            if ((numCounter % xFizzint == 0) && (numCounter % yBuzzint == 0)){
                System.out.println("FizzBuzz");
                numCounter++;
            }
            //This if statement prints FizzBuzz if the number is divisible by xFizzInt and yBuzzInt, and then adds to the counter.
            else if (numCounter % xFizzint == 0){
                System.out.println("Fizz");
                numCounter++;
            }
            //This if statement prints Fizz if the number is divisible by only xFizzInt, and then adds to the counter.
            else if (numCounter % yBuzzint == 0){
                System.out.println("Buzz");
                numCounter++;
            }
            //This if statement prints Buzz if the number is divisible by only yBuzzInt, and then adds to the counter.
            else{
                System.out.println(numCounter);
                numCounter++;
            }
            //This if statement prints the number if not divisible by xFizzInt or yBuzzInt, and then adds to the counter.
            //
            if (numCounter > nLimit){
                gameStillGoing = false;
            }
            //This if statement stops the code depending on the limit set by the user.
        }




    }
}
