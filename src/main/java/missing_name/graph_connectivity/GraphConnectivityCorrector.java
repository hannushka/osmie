package missing_name.graph_connectivity;

import org.openstreetmap.atlas.geography.atlas.items.Edge;
import util.HelperFunctions;
import util.SpellObject;

import java.util.*;

public class GraphConnectivityCorrector {

    public static void run(SpellObject so, Edge e) {
        if (HelperFunctions.isNoNameTagged(e))
            return;
        Optional<String> name = e.getTag("name");
        if (!name.isPresent() && !HelperFunctions.isRoundAbout(e)) {
            SmoothingCases.run(so, e);
        }
    }
}
