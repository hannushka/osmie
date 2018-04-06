package s_corrector.random_forest;

import org.apache.commons.io.IOUtils;
import org.openstreetmap.atlas.geography.atlas.items.Edge;
import org.openstreetmap.atlas.tags.names.NameTag;
import util.SpellObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class SCorrector {

    public static void run(SpellObject so, Edge edge) {
        Optional<String> nameTag = edge.getTag(NameTag.KEY);
        if (!nameTag.isPresent() || !nameTag.get().contains("s"))
            return;
        String name = nameTag.get().trim().toLowerCase();
        so.addName(name);
        List<String> commands = new ArrayList<>();
        commands.add("python3");
        commands.add("src/main/java/spellchecker/random_forest/s_corrector.py");
        commands.add(name);
        ProcessBuilder pb = new ProcessBuilder(commands);
        try {
            Process p = pb.start();
            p.waitFor();
            List<String> outputList = IOUtils.readLines(p.getInputStream(), "utf-8");
            if (outputList.size() < 1)
                throw new IOException("No output from S corrector.");
            String correctedName = outputList.get(1);
            so.addSuggestion(correctedName);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
