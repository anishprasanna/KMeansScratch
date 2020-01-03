import java.util.Random;
public class SumRandomIntegers {
    public static void main(String [] arg)
    {
        Random rnum = new Random();              //Set up random generator

        int randSum;                            //Variable for the sum of the 100 random integers
        double randAve;                        //Variable for the average of the 100 random integers
        int randMax;                          //Variable for the maximum value of the 100 random integers
        int randNum;                         //Variable for the random number generated at the time
        int i;                              //Variable for the number of times going through the for loop

        System.out.println("Random Integers =============");        //Header
        System.out.println("");                                    //Empty line

        randSum = 0;                                    //Assigning randSum to 0 to be changed later in loop
        randMax = 0;                                   //Assigning randMax to 0 to be changed in if else in for loop
        for (i =0; i < 100; ++i) {                    //Adding the random numbers together and finding the maximum
            randNum = rnum.nextInt(1000);     //Setting up random generator on range (0, 1000)
            randSum = randSum + randNum;
            if (randNum > randMax) {
                randMax = randNum;
            }
        }
        System.out.println("The sum of random 100 integers was " + randSum);        //Outputting to screen the sum

        randAve = randSum / 100;                                       //Equation to calculate the average of the 100 random integers
        System.out.printf("The average of 100 random numbers was %.2f\n" , randAve);        //Outputting to screen the average

        System.out.println("The largest number among 100 random integers was " + randMax);      //Outputting to screen the maximum

    }
}
