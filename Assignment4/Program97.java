import java.util.Scanner;

public class PredictGPA {
    public static void main(String [] arg){
        Scanner scnr = new Scanner(System.in);
        int creditHours;
        int courseGPATimesHours;
        String letterGrade;
        int totalCreditHours = 0;
        int totalGPATimesHours = 0;

        System.out.print("Course 1: Number of hours? ");
        creditHours = scnr.nextInt();
        System.out.print("Course 1: Expected grade? ");
        letterGrade = scnr.next();
        switch (letterGrade){
            case "A":
                courseGPATimesHours = 4 * creditHours;
                break;
            case "B":
                courseGPATimesHours = 3 * creditHours;
                break;
            case "C":
                courseGPATimesHours = 2 * creditHours;
                break;
            case "D":
                courseGPATimesHours = 1 * creditHours;
            case "a":
                courseGPATimesHours = 4 * creditHours;
                break;
            case "b":
                courseGPATimesHours = 3 * creditHours;
                break;
            case "c":
                courseGPATimesHours = 2 * creditHours;
                break;
            case "d":
                courseGPATimesHours = 1 * creditHours;
            default:
                courseGPATimesHours = 0;
                break;
        }
        totalGPATimesHours = totalGPATimesHours  + courseGPATimesHours;

        //totalCreditHours acts as a counter
        totalCreditHours = totalCreditHours + creditHours;

        System.out.print("Course 2: Number of hours? ");
        creditHours = scnr.nextInt();
        System.out.print("Course 2: Expected grade? ");
        letterGrade = scnr.next();
        switch (letterGrade) {
            case "A":
                courseGPATimesHours = 4 * creditHours;
                break;
            case "B":
                courseGPATimesHours = 3 * creditHours;
                break;
            case "C":
                courseGPATimesHours = 2 * creditHours;
                break;
            case "D":
                courseGPATimesHours = 1 * creditHours;
            case "a":
                courseGPATimesHours = 4 * creditHours;
                break;
            case "b":
                courseGPATimesHours = 3 * creditHours;
                break;
            case "c":
                courseGPATimesHours = 2 * creditHours;
                break;
            case "d":
                courseGPATimesHours = 1 * creditHours;
            default:
                courseGPATimesHours = 0;
                break;
        }
        totalGPATimesHours = totalGPATimesHours  + courseGPATimesHours;
        totalCreditHours = totalCreditHours + creditHours;


        System.out.print("Course 3: Number of hours? ");
        creditHours = scnr.nextInt();
        System.out.print("Course 3: Expected grade? ");
        letterGrade = scnr.next();
        switch (letterGrade) {
            case "A":
                courseGPATimesHours = 4 * creditHours;
                break;
            case "B":
                courseGPATimesHours = 3 * creditHours;
                break;
            case "C":
                courseGPATimesHours = 2 * creditHours;
                break;
            case "D":
                courseGPATimesHours = 1 * creditHours;
            case "a":
                courseGPATimesHours = 4 * creditHours;
                break;
            case "b":
                courseGPATimesHours = 3 * creditHours;
                break;
            case "c":
                courseGPATimesHours = 2 * creditHours;
                break;
            case "d":
                courseGPATimesHours = 1 * creditHours;
            default:
                courseGPATimesHours = 0;
                break;
        }
        totalGPATimesHours = totalGPATimesHours  + courseGPATimesHours;
        totalCreditHours = totalCreditHours + creditHours;


        System.out.print("Course 4: Number of hours? ");
        creditHours = scnr.nextInt();
        System.out.print("Course 4: Expected grade? ");
        letterGrade = scnr.next();
        switch (letterGrade) {
            case "A":
                courseGPATimesHours = 4 * creditHours;
                break;
            case "B":
                courseGPATimesHours = 3 * creditHours;
                break;
            case "C":
                courseGPATimesHours = 2 * creditHours;
                break;
            case "D":
                courseGPATimesHours = 1 * creditHours;
            case "a":
                courseGPATimesHours = 4 * creditHours;
                break;
            case "b":
                courseGPATimesHours = 3 * creditHours;
                break;
            case "c":
                courseGPATimesHours = 2 * creditHours;
                break;
            case "d":
                courseGPATimesHours = 1 * creditHours;
            default:
                courseGPATimesHours = 0;
                break;
        }
        totalGPATimesHours = totalGPATimesHours  + courseGPATimesHours;
        totalCreditHours = totalCreditHours + creditHours;

//        checking to see if counter works
//        System.out.print(totalCreditHours);

        System.out.print("Your semester GPA would be: " );
        System.out.printf("%.2f", (double)totalGPATimesHours/totalCreditHours);
    }
}
