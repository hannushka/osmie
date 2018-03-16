package missing_name.graph_connectivity;

import org.openstreetmap.atlas.geography.atlas.items.Edge;
import util.SpellObject;

import java.util.*;

public class GraphConnectivityCorrector {

    public static void run(SpellObject so, Edge e) {
        Optional<String> name = e.getTag("name");
        if (!name.isPresent() && !HelperFunctions.isRoundAbout(e)) {
            InBetweenNamedEdges.run(so, e);
        }
    }
}
