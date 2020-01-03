import java.util.Scanner;
public class OrderedIntegers {
    public static void main (String [] args)
    {
        Scanner scnr = new Scanner (System.in);

        //identify variables
        int integerA;
        int integerB;
        int integerC;
        // prompt user for data
        System.out.print("Enter three integers: ");
        integerA = scnr.nextInt();
        integerB = scnr.nextInt();
        integerC = scnr.nextInt();

        // order integers
        if (integerA > integerB && integerA > integerC && integerB > integerC) {
            System.out.println(integerA +  " " + integerB + " " + integerC);
        }
        else if (integerA > integerB && integerA > integerC && integerC > integerB) {
            System.out.println(integerA + " " + integerC + " " + integerB);
        }
        else if (integerB > integerA && integerB > integerC && integerC > integerA) {
            System.out.print(integerB + " " + integerC + " " + integerA);
        }
        else if (integerB > integerA && integerB > integerC && integerA > integerC) {
            System.out.print(integerB + " " + integerA + " " + integerC);
        }
        else if (integerC > integerA && integerC > integerB && integerA > integerB) {
            System.out.print(integerC + " " + integerA + " " + integerB);
        }
        else if (integerC > integerA && integerC > integerB && integerB > integerA) {
            System.out.print(integerC + " " + integerB + " " + integerA);
        }
        }


    }

