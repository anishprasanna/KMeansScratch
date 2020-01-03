import java.util.Scanner;
public class FizzBuzzGame {
    public static void main(String [] args){

        Scanner scnr = new Scanner(System.in);

        //Declare the variables
        int integerX;
        int integerY;
        int integerN;
        int counter;

        //Prompt the user to enter three integers
        System.out.println("Fizz Buzz Game =============");
        System.out.println();
        System.out.println("Input three integers: ");
        integerX = scnr.nextInt();
        integerY = scnr.nextInt();
        integerN = scnr.nextInt();

        //Use a loop to understand if each number from 1 to N is divisible for either X or Y or both. Print output accordingly
        for (counter = 1; counter <= integerN; counter++){
            if ( (counter % integerX) == 0  &&  (counter % integerY) == 0 ) {
                System.out.println("FizzBuzz");
            }
            else if ((counter % integerX) == 0 ){
                System.out.println("Fizz");
            }
            else if ((counter % integerY) == 0 ){
                System.out.println("Buzz");
            }
            else{
                System.out.println(counter);
            }
        }



    }
}
