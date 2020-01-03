import java.util.Scanner;

public class TaxDeductionWithNeaterLogic {
    public static void main(String [] arg) {
        Scanner scnr = new Scanner(System.in);
        String userItem;
        String yesNo;
        int taxDeduction = 2000;
        String firstLetter;


        System.out.print("Do you have any itemized deduction? [Y or N] ");
        yesNo = scnr.next();
        yesNo = yesNo.toLowerCase();
        if (!yesNo.equals("y")) {
            System.out.println("Please file with the standard deduction, which is $2000.");
            System.exit(0);
        }
        else {
            System.out.print("Enter the name of your itemized deduction: ");
            userItem = scnr.next();
            //this ensures that the characters in string are homogeneous
            //it will make it easier to manipulate string if input starts off at the same casing
            //this also means that the user can input any case and it does not matter
            userItem =userItem.toLowerCase();
            //using firstLetter variable to make the firdt letter capitalized
            firstLetter = String.valueOf(userItem.charAt(0));
            firstLetter = firstLetter.toUpperCase();

            switch (userItem){
                case "mortgage":
                case "property":
                case "investment":
                case "charitable":
                case "medical":
                case "sales":
                    userItem = userItem.substring(1, userItem.length());
                    System.out.print("Enter the amount of deduction for " + firstLetter + userItem + ": ");
                    taxDeduction = scnr.nextInt();
                    break;
                default:
                    System.out.println(userItem + " may not be deductible.");
                    System.out.print("Please file with the itemized deduction, which is $2000.");
                    System.exit(0);
            }
//        char userItemchar = userItem.charAt(0);


            System.out.print("Please file with the itemized deduction: " + firstLetter + userItem + ", which is $" + taxDeduction);
        }
    }
}
