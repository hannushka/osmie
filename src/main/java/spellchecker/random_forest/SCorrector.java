package spellchecker.random_forest;

import org.apache.commons.io.IOUtils;
import util.SpellObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class SCorrector {

    public void run(SpellObject so) {
        if (!so.name.contains("s"))
            return;
        List<String> commands = new ArrayList<>();
        commands.add("python3");
        commands.add("src/main/java/spellchecker/random_forest/s_corrector.py");
        commands.add(so.name.trim().toLowerCase());
        ProcessBuilder pb = new ProcessBuilder(commands);
        try {
            Process p = pb.start();
            p.waitFor();
            List<String> outputList = IOUtils.readLines(p.getInputStream(), "utf-8");
            if (outputList.size() < 1)
                throw new IOException("No output from S corrector.");
            String correctedName = outputList.get(1);
            so.addCorrection(correctedName);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 1 || args[0].isEmpty())
            throw new IOException("No String argument to Spellcorrector.");
        SpellObject so = new SpellObject(args[0]);
        new SCorrector().run(so);
        so.print();
    }
}
