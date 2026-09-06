import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../../models/inspection.dart';
import '../../services/api_config.dart';
import '../../theme.dart';
import '../../widgets/status_badge.dart';
import '../../widgets/authenticated_image.dart';

const _clsLabel = {
  'material_loss': 'Material Loss',
  'peel': 'Peeling',
  'scratch': 'Scratch',
  'writing_marks': 'Writing Marks',
  'dirt': 'Dirt',
  'staning': 'Staining',
  'staining': 'Staining',
  'burn_marks': 'Burn Marks',
};

class _SsimGrade {
  final String label;
  final String description;
  final Color color;
  final IconData icon;

  const _SsimGrade(this.label, this.description, this.color, this.icon);

  static _SsimGrade from(double ssim) {
    if (ssim >= 0.95) {
      return const _SsimGrade(
        'Excellent',
        'The artifact closely matches the reference image.',
        AppColors.statusGood,
        Icons.verified_outlined,
      );
    }
    if (ssim >= 0.85) {
      return const _SsimGrade(
        'Good',
        'Only minor visible differences were found.',
        Colors.lightGreen,
        Icons.thumb_up_outlined,
      );
    }
    if (ssim >= 0.70) {
      return const _SsimGrade(
        'Fair',
        'Some visible differences may need review.',
        Colors.orange,
        Icons.warning_amber_outlined,
      );
    }
    if (ssim >= 0.50) {
      return const _SsimGrade(
        'Poor',
        'Clear visible differences were found.',
        Colors.deepOrange,
        Icons.report_outlined,
      );
    }
    return const _SsimGrade(
      'Critical',
      'Major visible changes were found compared with the reference.',
      AppColors.statusDamaged,
      Icons.dangerous_outlined,
    );
  }
}

class ResultScreen extends StatelessWidget {
  final Inspection inspection;

  const ResultScreen({super.key, required this.inspection});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Inspection Result'),
          actions: [
            Padding(
              padding: const EdgeInsets.only(right: 12),
              child: StatusBadge(status: inspection.status),
            ),
          ],
          bottom: const TabBar(
            labelColor: Colors.white,
            unselectedLabelColor: Colors.white60,
            indicatorColor: Colors.white,
            indicatorWeight: 3,
            tabs: [
              Tab(icon: Icon(Icons.compare_outlined), text: 'Visual Review'),
              Tab(icon: Icon(Icons.manage_search_outlined), text: 'Damage Detection'),
            ],
          ),
        ),
        body: SafeArea(
          child: TabBarView(
            children: [
              _VisualReviewTab(inspection: inspection),
              _DetectTab(inspection: inspection),
            ],
          ),
        ),
      ),
    );
  }
}

class _VisualReviewTab extends StatelessWidget {
  final Inspection inspection;

  const _VisualReviewTab({required this.inspection});

  @override
  Widget build(BuildContext context) {
    final ssim = inspection.ssimValue;
    final damagePct = inspection.ssimDamagePct;

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        if (ssim != null) ...[
          _buildGradeCard(ssim, damagePct),
          const SizedBox(height: 16),
          _buildSimilarityDetails(ssim),
          const SizedBox(height: 16),
        ] else
          _buildNoReferenceCard(),
        _ImageCard(
          title: 'Difference Heatmap',
          subtitle: 'Warm colors show stronger visible differences. Cool colors show closer matches.',
          icon: Icons.thermostat_outlined,
          url: inspection.heatmapPath != null
              ? ApiConfig.resolveAssetUrl(inspection.heatmapPath)
              : null,
          emptyLabel: inspection.heatmapPath == null
              ? 'No heatmap available\nInitialize Golden Pose first.'
              : 'Could not load heatmap',
        ),
        const SizedBox(height: 16),
        _buildImageComparison(context),
        const SizedBox(height: 16),
        _buildMetaCard(),
        const SizedBox(height: 24),
      ],
    );
  }

  Widget _buildGradeCard(double ssim, double? damagePct) {
    final grade = _SsimGrade.from(ssim);
    final pct = ssim * 100;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: grade.color.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: grade.color.withValues(alpha: 0.35), width: 1.5),
      ),
      child: Column(
        children: [
          Row(
            children: [
              SizedBox(
                width: 72,
                height: 72,
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    CircularProgressIndicator(
                      value: ssim.clamp(0.0, 1.0),
                      strokeWidth: 6,
                      backgroundColor: grade.color.withValues(alpha: 0.15),
                      valueColor: AlwaysStoppedAnimation<Color>(grade.color),
                    ),
                    Text(
                      '${pct.toStringAsFixed(1)}%',
                      style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.bold,
                        color: grade.color,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(grade.icon, size: 18, color: grade.color),
                        const SizedBox(width: 6),
                        Expanded(
                          child: Text(
                            'Similarity Rating: ${grade.label}',
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 16,
                              color: grade.color,
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Text(
                      grade.description,
                      style: const TextStyle(
                        fontSize: 12,
                        color: AppColors.textMuted,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          if (damagePct != null) ...[
            const SizedBox(height: 14),
            const Divider(height: 1),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _StatItem(
                  label: 'Similarity',
                  value: '${pct.toStringAsFixed(1)}%',
                  color: grade.color,
                  tooltip: 'How closely this image matches the reference.',
                ),
                _StatItem(
                  label: 'Changed Area',
                  value: '${damagePct.toStringAsFixed(1)}%',
                  color: damagePct > 10 ? Colors.orange : AppColors.statusGood,
                  tooltip: 'Visible area that differs from the reference.',
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildSimilarityDetails(double ssim) {
    final summary = inspection.ssimSummary;
    final grayVal = (summary?['ssim_gray'] as num?)?.toDouble();
    final colorVal = (summary?['ssim_color'] as num?)?.toDouble();

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.surfaceMuted,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.bar_chart, size: 16, color: AppColors.primary),
              SizedBox(width: 6),
              Text(
                'Similarity Details',
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 13),
              ),
            ],
          ),
          const SizedBox(height: 12),
          _SsimBar(
            label: 'Overall Similarity',
            value: ssim,
            color: _SsimGrade.from(ssim).color,
            bold: true,
          ),
          if (grayVal != null) ...[
            const SizedBox(height: 8),
            _SsimBar(
              label: 'Shape & Detail Match',
              value: grayVal,
              color: Colors.blueGrey,
            ),
          ],
          if (colorVal != null) ...[
            const SizedBox(height: 8),
            _SsimBar(
              label: 'Color Match',
              value: colorVal,
              color: Colors.purple,
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildNoReferenceCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      margin: const EdgeInsets.only(bottom: 16),
      decoration: BoxDecoration(
        color: Colors.grey.shade100,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.grey.shade300),
      ),
      child: const Row(
        children: [
          Icon(Icons.info_outline, color: AppColors.textMuted),
          SizedBox(width: 12),
          Expanded(
            child: Text(
              'Visual review requires a Golden Pose reference image.\n'
              'Initialize Golden Pose first to compare this inspection with the reference.',
              style: TextStyle(fontSize: 13, color: AppColors.textMuted),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildImageComparison(BuildContext context) {
    final referencePath = inspection.previousImagePath;
    final hasReferenceImage = referencePath != null &&
        referencePath.isNotEmpty &&
        referencePath != inspection.currentImagePath;
    final referenceUrl = hasReferenceImage
        ? ApiConfig.resolveAssetUrl(referencePath)
        : null;
    final alignedUrl = inspection.alignedImagePath != null
        ? ApiConfig.resolveAssetUrl(inspection.alignedImagePath)
        : null;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Row(
          children: [
            Icon(Icons.compare, size: 16, color: AppColors.primary),
            SizedBox(width: 6),
            Text(
              'Image Comparison',
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 13),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: _MiniImageTile(
                label: 'Reference Image',
                url: referenceUrl,
                emptyIcon: Icons.image_search_outlined,
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _MiniImageTile(
                label: 'Aligned Inspection',
                url: alignedUrl,
                emptyIcon: Icons.align_horizontal_center_outlined,
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildMetaCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          children: [
            _MetaRow(
              label: 'Inspection Type',
              value: inspection.inspectionType == InspectionType.scheduled
                  ? 'Scheduled'
                  : 'Ad-hoc',
              valueColor: inspection.inspectionType == InspectionType.scheduled
                  ? Colors.blue
                  : Colors.orange,
            ),
            const Divider(height: 22),
            _MetaRow(
              label: 'Date',
              value: DateFormat('HH:mm dd/MM/yyyy').format(inspection.createdAt),
            ),
            if (inspection.description.isNotEmpty) ...[
              const Divider(height: 22),
              _MetaRow(label: 'Note', value: inspection.description),
            ],
          ],
        ),
      ),
    );
  }
}

class _DetectTab extends StatelessWidget {
  final Inspection inspection;

  const _DetectTab({required this.inspection});

  @override
  Widget build(BuildContext context) {
    final detections = inspection.detections;
    final detectionImageUrl = inspection.annotatedImagePath != null
        ? ApiConfig.resolveAssetUrl(inspection.annotatedImagePath)
        : null;
    final detectionImageSize = inspection.detectionImageSize;

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        _ImageCard(
          title: 'Detected Damage',
          subtitle: 'Highlighted areas show possible damage found in this inspection.',
          icon: Icons.manage_search_outlined,
          url: inspection.annotatedImagePath != null
              ? ApiConfig.resolveAssetUrl(inspection.annotatedImagePath)
              : null,
          emptyLabel: 'No annotated image available.',
        ),
        const SizedBox(height: 20),
        _buildDetectionList(
          context,
          detections,
          detectionImageUrl,
          detectionImageSize,
        ),
        const SizedBox(height: 24),
      ],
    );
  }

  Widget _buildDetectionList(
    BuildContext context,
    List<Map<String, dynamic>> detections,
    String? detectionImageUrl,
    List<double>? detectionImageSize,
  ) {
    if (detections.isEmpty) {
      return Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: AppColors.statusGood.withValues(alpha: 0.08),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppColors.statusGood.withValues(alpha: 0.3)),
        ),
        child: const Row(
          children: [
            Icon(Icons.check_circle_outline, color: AppColors.statusGood, size: 22),
            SizedBox(width: 10),
            Expanded(
              child: Text(
                'No damage detected',
                style: TextStyle(
                  color: AppColors.statusGood,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          '${detections.length} damage region(s) detected',
          style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
        ),
        const SizedBox(height: 10),
        ...detections.map((detection) {
          final rawName   = detection['class_name']?.toString() ?? '';
          final name      = _clsLabel[rawName] ?? rawName;
          final confidence =
              ((detection['confidence'] as num?)?.toDouble() ?? 0.0) * 100;
          final regionSsim =
              (detection['region_ssim'] as num?)?.toDouble();
          final bbox = _bboxFromDetection(detection);
          final confColor = confidence >= 65
              ? AppColors.statusDamaged
              : confidence >= 40
                  ? AppColors.statusWarning
                  : AppColors.statusNeedCheck;

          return Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
              decoration: BoxDecoration(
                color: confColor.withValues(alpha: 0.07),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: confColor.withValues(alpha: 0.25)),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.warning_amber_rounded, size: 18, color: confColor),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          name.isNotEmpty ? name : 'Unknown',
                          style: const TextStyle(fontWeight: FontWeight.w500),
                        ),
                        const SizedBox(height: 3),
                        if (regionSsim != null)
                          Text(
                            'Region Similarity: ${(regionSsim * 100).toStringAsFixed(1)}%',
                            style: const TextStyle(
                              fontSize: 11,
                              color: AppColors.textMuted,
                            ),
                          ),
                        Text(
                          'AI Confidence: ${confidence.toStringAsFixed(1)}%',
                          style: TextStyle(
                            fontSize: 11,
                            color: confColor,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 10),
                  _DetectionBboxThumb(
                    imageUrl: detectionImageUrl,
                    imageSize: detectionImageSize,
                    bbox: bbox,
                    title: name.isNotEmpty ? '$name region' : 'Detection region',
                  ),
                ],
              ),
            ),
          );
        }),
      ],
    );
  }
}

class _BboxRect {
  final double x1;
  final double y1;
  final double x2;
  final double y2;

  const _BboxRect(this.x1, this.y1, this.x2, this.y2);

  double get left => math.min(x1, x2);
  double get top => math.min(y1, y2);
  double get right => math.max(x1, x2);
  double get bottom => math.max(y1, y2);
  double get width => math.max(1, right - left);
  double get height => math.max(1, bottom - top);
  double get centerX => left + width / 2;
  double get centerY => top + height / 2;
  double get aspectRatio => width / height;

  bool get isValid => width > 1 && height > 1;

  _BboxRect clamped(double imageWidth, double imageHeight) {
    final nx1 = left.clamp(0.0, imageWidth);
    final ny1 = top.clamp(0.0, imageHeight);
    final nx2 = right.clamp(0.0, imageWidth);
    final ny2 = bottom.clamp(0.0, imageHeight);
    return _BboxRect(nx1, ny1, nx2, ny2);
  }
}

_BboxRect? _bboxFromDetection(Map<String, dynamic> detection) {
  final raw = detection['bbox_xyxy'];
  if (raw is! List || raw.length < 4) return null;
  final x1 = (raw[0] as num?)?.toDouble();
  final y1 = (raw[1] as num?)?.toDouble();
  final x2 = (raw[2] as num?)?.toDouble();
  final y2 = (raw[3] as num?)?.toDouble();
  if (x1 == null || y1 == null || x2 == null || y2 == null) return null;
  final bbox = _BboxRect(x1, y1, x2, y2);
  return bbox.isValid ? bbox : null;
}

class _DetectionBboxThumb extends StatelessWidget {
  final String? imageUrl;
  final List<double>? imageSize;
  final _BboxRect? bbox;
  final String title;

  const _DetectionBboxThumb({
    required this.imageUrl,
    required this.imageSize,
    required this.bbox,
    required this.title,
  });

  @override
  Widget build(BuildContext context) {
    if (imageUrl == null || imageUrl!.isEmpty || bbox == null) {
      return _bboxPlaceholder();
    }

    return GestureDetector(
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => _BboxRegionPage(
            imageUrl: imageUrl!,
            imageSize: imageSize,
            bbox: bbox!,
            title: title,
          ),
          fullscreenDialog: true,
        ),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(8),
        child: SizedBox(
          width: 92,
          height: 74,
          child: Stack(
            fit: StackFit.expand,
            children: [
              _BboxRegionImage(
                imageUrl: imageUrl!,
                imageSize: imageSize,
                bbox: bbox!,
              ),
              Positioned(
                right: 4,
                bottom: 4,
                child: Container(
                  padding: const EdgeInsets.all(3),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.55),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: const Icon(
                    Icons.zoom_out_map,
                    color: Colors.white,
                    size: 13,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _BboxRegionImage extends StatefulWidget {
  final String imageUrl;
  final List<double>? imageSize;
  final _BboxRect bbox;

  const _BboxRegionImage({
    required this.imageUrl,
    required this.imageSize,
    required this.bbox,
  });

  @override
  State<_BboxRegionImage> createState() => _BboxRegionImageState();
}

class _BboxRegionImageState extends State<_BboxRegionImage> {
  ImageStream? _imageStream;
  ImageStreamListener? _imageListener;
  List<double>? _resolvedImageSize;

  List<double>? get _effectiveImageSize =>
      widget.imageSize ?? _resolvedImageSize;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _resolveImageSizeIfNeeded();
  }

  @override
  void didUpdateWidget(covariant _BboxRegionImage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.imageUrl != widget.imageUrl ||
        oldWidget.imageSize != widget.imageSize) {
      _removeImageListener();
      _resolvedImageSize = null;
      _resolveImageSizeIfNeeded();
    }
  }

  @override
  void dispose() {
    _removeImageListener();
    super.dispose();
  }

  void _resolveImageSizeIfNeeded() {
    if (widget.imageSize != null || _resolvedImageSize != null) return;

    final stream = NetworkImage(widget.imageUrl).resolve(
      createLocalImageConfiguration(context),
    );
    final listener = ImageStreamListener((info, _) {
      if (!mounted) return;
      setState(() {
        _resolvedImageSize = [
          info.image.width.toDouble(),
          info.image.height.toDouble(),
        ];
      });
    });

    _imageStream = stream;
    _imageListener = listener;
    stream.addListener(listener);
  }

  void _removeImageListener() {
    final stream = _imageStream;
    final listener = _imageListener;
    if (stream != null && listener != null) {
      stream.removeListener(listener);
    }
    _imageStream = null;
    _imageListener = null;
  }

  @override
  Widget build(BuildContext context) {
    final size = _effectiveImageSize;
    if (size == null || size.length < 2 || size[0] <= 0 || size[1] <= 0) {
      return Container(
        color: AppColors.surfaceMuted,
        child: const Center(
          child: CircularProgressIndicator(strokeWidth: 2),
        ),
      );
    }

    final imageWidth = size[0];
    final imageHeight = size[1];
    final bbox = widget.bbox.clamped(imageWidth, imageHeight);
    if (!bbox.isValid) return _bboxPlaceholder();

    return LayoutBuilder(
      builder: (context, constraints) {
        final viewWidth = constraints.maxWidth.isFinite
            ? constraints.maxWidth
            : 120.0;
        final viewHeight = constraints.maxHeight.isFinite
            ? constraints.maxHeight
            : 90.0;
        final scale = math.max(viewWidth / bbox.width, viewHeight / bbox.height);
        final left = viewWidth / 2 - bbox.centerX * scale;
        final top = viewHeight / 2 - bbox.centerY * scale;

        return ClipRect(
          child: Stack(
            children: [
              Positioned(
                left: left,
                top: top,
                width: imageWidth * scale,
                height: imageHeight * scale,
                child: AuthenticatedImage(
                  widget.imageUrl,
                  fit: BoxFit.fill,
                  errorBuilder: (_, __, ___) => _bboxPlaceholder(),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _BboxRegionPage extends StatelessWidget {
  final String imageUrl;
  final List<double>? imageSize;
  final _BboxRect bbox;
  final String title;

  const _BboxRegionPage({
    required this.imageUrl,
    required this.imageSize,
    required this.bbox,
    required this.title,
  });

  @override
  Widget build(BuildContext context) {
    final aspectRatio = bbox.aspectRatio.clamp(0.35, 2.8);

    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: Text(
          title,
          style: const TextStyle(fontSize: 14, color: Colors.white),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.close, color: Colors.white),
            onPressed: () => Navigator.pop(context),
          ),
        ],
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          final maxWidth = constraints.maxWidth;
          final maxHeight = constraints.maxHeight;
          var width = maxWidth;
          var height = width / aspectRatio;
          if (height > maxHeight) {
            height = maxHeight;
            width = height * aspectRatio;
          }

          return Center(
            child: InteractiveViewer(
              minScale: 0.8,
              maxScale: 8.0,
              child: SizedBox(
                width: width,
                height: height,
                child: _BboxRegionImage(
                  imageUrl: imageUrl,
                  imageSize: imageSize,
                  bbox: bbox,
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

Widget _bboxPlaceholder() {
  return ClipRRect(
    borderRadius: BorderRadius.circular(8),
    child: Container(
      width: 92,
      height: 74,
      color: AppColors.surfaceMuted,
      child: const Center(
        child: Icon(
          Icons.crop_free,
          color: AppColors.textFaint,
          size: 24,
        ),
      ),
    ),
  );
}

class _ImageCard extends StatelessWidget {
  final String title;
  final String? subtitle;
  final IconData icon;
  final String? url;
  final String? emptyLabel;

  const _ImageCard({
    required this.title,
    required this.icon,
    this.subtitle,
    this.url,
    this.emptyLabel,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(icon, size: 15, color: AppColors.primary),
            const SizedBox(width: 6),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 13,
                    ),
                  ),
                  if (subtitle != null)
                    Text(
                      subtitle!,
                      style: const TextStyle(
                        fontSize: 11,
                        color: AppColors.textFaint,
                      ),
                    ),
                ],
              ),
            ),
            const Icon(Icons.pinch_outlined, size: 13, color: AppColors.textFaint),
            const SizedBox(width: 3),
            const Text(
              'zoom',
              style: TextStyle(fontSize: 10, color: AppColors.textFaint),
            ),
          ],
        ),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(12),
          child: Container(
            constraints: const BoxConstraints(minHeight: 180, maxHeight: 380),
            width: double.infinity,
            color: AppColors.surfaceMuted,
            child: url != null && url!.isNotEmpty
                ? _ZoomableNetworkImage(url: url!, title: title)
                : _imagePlaceholder(
                    Icons.image_not_supported_outlined,
                    label: emptyLabel ?? 'Image unavailable',
                  ),
          ),
        ),
      ],
    );
  }
}

class _MiniImageTile extends StatelessWidget {
  final String label;
  final String? url;
  final IconData emptyIcon;

  const _MiniImageTile({
    required this.label,
    this.url,
    this.emptyIcon = Icons.image_not_supported_outlined,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 11,
            color: AppColors.textMuted,
            fontWeight: FontWeight.w600,
          ),
        ),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(10),
          child: AspectRatio(
            aspectRatio: 4 / 3,
            child: url != null && url!.isNotEmpty
                ? GestureDetector(
                    onTap: () => Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (_) => _FullScreenImagePage(
                          url: url!,
                          title: label,
                        ),
                        fullscreenDialog: true,
                      ),
                    ),
                    child: AuthenticatedImage(
                      url!,
                      fit: BoxFit.cover,
                      loadingBuilder: (_, child, progress) => progress == null
                          ? child
                          : const Center(child: CircularProgressIndicator()),
                      errorBuilder: (_, __, ___) => Container(
                        color: AppColors.surfaceMuted,
                        child: Center(
                          child: Icon(
                            emptyIcon,
                            color: AppColors.textFaint,
                            size: 30,
                          ),
                        ),
                      ),
                    ),
                  )
                : Container(
                    color: AppColors.surfaceMuted,
                    child: Center(
                      child: Icon(
                        emptyIcon,
                        color: AppColors.textFaint,
                        size: 30,
                      ),
                    ),
                  ),
          ),
        ),
      ],
    );
  }
}

class _ZoomableNetworkImage extends StatelessWidget {
  final String url;
  final String title;

  const _ZoomableNetworkImage({required this.url, this.title = 'Image'});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => _FullScreenImagePage(url: url, title: title),
          fullscreenDialog: true,
        ),
      ),
      child: AuthenticatedImage(
        url,
        fit: BoxFit.contain,
        width: double.infinity,
        loadingBuilder: (_, child, progress) => progress == null
            ? child
            : const Center(child: CircularProgressIndicator()),
        errorBuilder: (_, __, ___) =>
            _imagePlaceholder(Icons.broken_image_outlined),
      ),
    );
  }
}

class _FullScreenImagePage extends StatelessWidget {
  final String url;
  final String title;

  const _FullScreenImagePage({required this.url, required this.title});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: Text(
          title,
          style: const TextStyle(fontSize: 14, color: Colors.white),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.close, color: Colors.white),
            onPressed: () => Navigator.pop(context),
          ),
        ],
      ),
      body: Center(
        child: InteractiveViewer(
          minScale: 0.5,
          maxScale: 8.0,
          child: AuthenticatedImage(
            url,
            fit: BoxFit.contain,
            loadingBuilder: (_, child, progress) => progress == null
                ? child
                : const Center(
                    child: CircularProgressIndicator(color: Colors.white),
                  ),
            errorBuilder: (_, __, ___) => const Center(
              child: Icon(
                Icons.broken_image_outlined,
                color: Colors.white54,
                size: 64,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _SsimBar extends StatelessWidget {
  final String label;
  final double value;
  final Color color;
  final bool bold;

  const _SsimBar({
    required this.label,
    required this.value,
    required this.color,
    this.bold = false,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(
                label,
                style: TextStyle(
                  fontSize: 11,
                  color: AppColors.textMuted,
                  fontWeight: bold ? FontWeight.bold : FontWeight.normal,
                ),
              ),
            ),
            Text(
              '${(value * 100).toStringAsFixed(1)}%',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: value.clamp(0.0, 1.0),
            minHeight: bold ? 8 : 5,
            backgroundColor: color.withValues(alpha: 0.15),
            valueColor: AlwaysStoppedAnimation<Color>(color),
          ),
        ),
      ],
    );
  }
}

class _StatItem extends StatelessWidget {
  final String label;
  final String value;
  final Color color;
  final String tooltip;

  const _StatItem({
    required this.label,
    required this.value,
    required this.color,
    required this.tooltip,
  });

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: tooltip,
      child: Column(
        children: [
          Text(
            value,
            style: TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            label,
            style: const TextStyle(fontSize: 11, color: AppColors.textMuted),
          ),
        ],
      ),
    );
  }
}

class _MetaRow extends StatelessWidget {
  final String label;
  final String value;
  final Color? valueColor;

  const _MetaRow({required this.label, required this.value, this.valueColor});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: const TextStyle(
            color: AppColors.textMuted,
            fontWeight: FontWeight.w500,
          ),
        ),
        Flexible(
          child: Text(
            value,
            textAlign: TextAlign.end,
            style: TextStyle(
              fontWeight: FontWeight.w600,
              color: valueColor,
            ),
          ),
        ),
      ],
    );
  }
}

Widget _imagePlaceholder(IconData icon, {String? label}) {
  return Container(
    color: AppColors.surfaceMuted,
    child: Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 40, color: AppColors.textFaint),
          if (label != null)
            Padding(
              padding: const EdgeInsets.only(top: 8, left: 16, right: 16),
              child: Text(
                label,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 11,
                  color: AppColors.textFaint,
                ),
              ),
            ),
        ],
      ),
    ),
  );
}
