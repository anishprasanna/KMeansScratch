import java.util.Random;
import java.util.Scanner;

public class FizzBuzzGame {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        Random rand = new Random();
        int x = 0, y = 0, n =0, i;
        int int1, int2, int3;

        //Fizz Buzz Game =============
        System.out.println("Fizz Buzz Game =============\n");

        //Ask user to input three integers: (1 <= X < Y < N <= 100)
        int1 = rand.nextInt(100) + 1;
        int2 = rand.nextInt(100) + 1;
        int3 = rand.nextInt(100) + 1;

        //System.out.println(int1 + " " + int2 + " " + int3);



        while (x == y) {
            if (int2 >= int1 && int2 >= int3 && int1 != int3) { // int2 is max; making sure x!=y, y can = n
                n = int2;
                if (int1 > int3) { // int1 is 2nd
                    y = int1;
                    x = int3;
                } else { // int 3 is 2nd
                    x = int1;
                    y = int3;
                }
            } else if (int3 >= int1 && int3 >= int2 && int2 != int1) { // int3 is max
                n = int3;
                if (int1 > int2) { // int1 is 2nd
                    y = int1;
                    x = int2;
                } else { // int 2 is 2nd
                    x = int1;
                    y = int2;
                }
            } else if (int1 >= int2 && int1 >= int3 && int2 != int3) { // int1 is max
                n = int1;
                if (int2 > int3) { // int2 is 2nd
                    y = int2;
                    x = int3;
                } else { // int 3 is 2nd
                    x = int2;
                    y = int3;
                }
            } else {
                int1 = rand.nextInt(100) + 1;
                int2 = rand.nextInt(100) + 1;
                int3 = rand.nextInt(100) + 1;
            }
        }
        System.out.println("Input three integers: " + x + " " + y + " " + n);


        //Print integers from 1 to N in order, each on its own line
        // replacing the ones divisible by X with Fizz; the ones divisible by Y with Buzz; Ones divisible by both X and Y with FizzBuzz.

        for (i=1; i<=n; i++){
            if (i % x == 0 && i % y == 0){
                System.out.println("FizzBuzz");
            }
            else if (i % x == 0){
                System.out.println("Fizz");
            }
            else if (i % y == 0){
                System.out.println("Buzz");
            }
            else{
                System.out.println(i);
            }
        }

    }
}

