import java.util.Scanner;
public class TaxDeduction {
    public static void main(String[] args) {
        Scanner scnr = new Scanner(System.in);

        //Declare variables
        char itemizedDeduction;
        String itemizedDeductionName;
        Double itemizedDeductionAmount;
        final int STANDARD_DEDUCTION = 2000;

        //Ask whether user has an itemized deduction or not and record the answer
        System.out.println("Tax Deduction ==========");
        System.out.println();
        System.out.print("Do you have any itemized deduction? [Y or N] ");
        itemizedDeduction = scnr.next().charAt(0);

        //If the user doesn't have any itemized deduction print to file with standard deduction [got the idea of using Character.toUpperCase from way2java.com)
        if (Character.toUpperCase(itemizedDeduction) == 'N') {
            System.out.println("Please file with the standard deduction, which is $2000.");
        }

        //If the user does have itemized deduction, ask for the name
        else if (Character.toUpperCase(itemizedDeduction) == 'Y') {
            System.out.println("Enter the name of your itemized deduction: ");
            itemizedDeductionName = scnr.next();

            //If name is within the 6 options, ask for amount and then compute if it's worth to file itemized deduction
            if ((itemizedDeductionName.equalsIgnoreCase("Mortgage")) || (itemizedDeductionName.equalsIgnoreCase("Property")) || (itemizedDeductionName.equalsIgnoreCase("Medical")) || (itemizedDeductionName.equalsIgnoreCase("Charitable")) || (itemizedDeductionName.equalsIgnoreCase("Investment")) || (itemizedDeductionName.equalsIgnoreCase("Sales"))) {
                System.out.println("Enter the amount of deduction for Mortgage: ");
                itemizedDeductionAmount = scnr.nextDouble();
                if (itemizedDeductionAmount > STANDARD_DEDUCTION) {
                    System.out.println("Please file with itemized deduction: " + itemizedDeductionName + ", which is $" + itemizedDeductionAmount + ".");
                } else {
                    System.out.println("Please file with the standard deduction, which is $2000.");
                }
            }
            else{
                    System.out.println("" + itemizedDeductionName + "might not be deductible.");
                    System.out.println("Please file with the standard deduction, which is $2000.");
                }

        }


    }
}

