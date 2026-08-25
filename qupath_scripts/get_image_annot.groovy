import qupath.lib.objects.PathObjects
import qupath.lib.roi.RoiTools

def imageData = getCurrentImageData()

// -------------------------
def hierarchy = imageData.getHierarchy()

// 1. Get all annotations and sort by their internal ID (Creation Order)
def annotations = getAnnotationObjects().sort { it.getID() }

// 2. Remove them and immediately re-add them in order
// This ensures the internal list QuPath maintains is ordered 1st to last
hierarchy.removeObjects(annotations, true)
hierarchy.addObjects(annotations)

// 3. Force a hierarchy update to settle the changes
fireHierarchyUpdate()
// -------------------------

// Define output path (relative to project)
def outputDir = buildFilePath(PROJECT_BASE_DIR, 'export')
mkdirs(outputDir)
def name = GeneralTools.stripExtension(imageData.getServer().getMetadata().getName())
def path = buildFilePath(outputDir, name + "-labels.tif")

// Define how much to downsample during export (may be required for large images)
double downsample = 64

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
  .backgroundLabel(0, ColorTools.WHITE) // Specify background label (usually 0 or 255)
  .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
  .addLabel('Tumor', 1)      // Choose output labels (the order matters!)
  .addLabel('Stroma', 2)
  .addLabel('Necrosis', 3)
  .addLabel('Other', 4)
  .multichannelOutput(true) // If true, each label refers to the channel of a multichannel binary image (required for multiclass probability)
  .build()

// Write the image
writeImage(labelServer, path)