import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';

import '../../models/artifact.dart';
import '../../models/artifact_status.dart';
import '../../models/inspection.dart';
import '../../providers/artifact_provider.dart';
import '../../services/api_config.dart';
import '../../services/token_storage.dart';
import '../../theme.dart';
import '../../widgets/responsive_scaffold.dart';
import '../../widgets/status_badge.dart';
import '../../widgets/authenticated_image.dart';
import '../devices/device_workflow_screen.dart';
import '../inspect/result_screen.dart';
import 'edit_artifact_screen.dart';

class ArtifactDetailScreen extends StatelessWidget {
  final Artifact artifact;

  const ArtifactDetailScreen({super.key, required this.artifact});

  @override
  Widget build(BuildContext context) {
    return _ArtifactDetailContent(artifact: artifact);
  }
}

class _ArtifactDetailContent extends StatefulWidget {
  final Artifact artifact;
  const _ArtifactDetailContent({required this.artifact});

  @override
  State<_ArtifactDetailContent> createState() => _ArtifactDetailContentState();
}

class _ArtifactDetailContentState extends State<_ArtifactDetailContent> {
  bool _busy = false;
  late Artifact _artifact;
  String? _userRole;

  List<Inspection> _inspections = [];
  bool _historyLoading = true;
  bool _historyExpanded = true;

  @override
  void initState() {
    super.initState();
    _artifact = widget.artifact;
    _loadRole();
    _loadHistory();
  }

  Future<void> _loadRole() async {
    final role = await context.read<TokenStorage>().readRole();
    if (mounted) setState(() => _userRole = role);
  }

  Future<void> _loadHistory() async {
    setState(() => _historyLoading = true);
    try {
      final list = await context.read<ArtifactProvider>().historyFor(_artifact.id);
      if (mounted) setState(() { _inspections = list; _historyLoading = false; });
    } catch (_) {
      if (mounted) setState(() => _historyLoading = false);
    }
  }

  Future<void> _runInspection() async {
    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => DeviceWorkflowScreen(preselectedArtifactId: _artifact.id),
      ),
    );
    if (mounted) _loadHistory();
  }

  Future<void> _editArtifact() async {
    final updated = await Navigator.push<bool>(
      context,
      MaterialPageRoute(builder: (_) => EditArtifactScreen(artifact: _artifact)),
    );
    if (updated == true && mounted) {
      await context.read<ArtifactProvider>().refresh();
      final refreshed = context.read<ArtifactProvider>().artifacts
          .firstWhere((a) => a.id == _artifact.id, orElse: () => _artifact);
      setState(() => _artifact = refreshed);
    }
  }

  Future<void> _archiveArtifact() async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Archive Artifact'),
        content: Text('Archive "${_artifact.name}"?'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: TextButton.styleFrom(foregroundColor: Colors.orange),
            child: const Text('Archive'),
          ),
        ],
      ),
    );
    if (confirm == true && mounted) {
      setState(() => _busy = true);
      try {
        await context.read<ArtifactProvider>().updateStatus(_artifact.id, ArtifactStatus.archived);
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Artifact archived')));
          Navigator.pop(context);
        }
      } catch (e) {
        if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
      } finally {
        if (mounted) setState(() => _busy = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final isAdmin = _userRole == 'admin';
    return Scaffold(
      appBar: AppBar(
        title: Text(_artifact.name),
        actions: [
          IconButton(tooltip: 'Edit', icon: const Icon(Icons.edit_outlined), onPressed: _editArtifact),
          if (isAdmin)
            IconButton(
              tooltip: 'Archive',
              icon: const Icon(Icons.archive_outlined, color: Colors.orangeAccent),
              onPressed: _archiveArtifact,
            ),
        ],
      ),
      body: SafeArea(
        child: ResponsiveBody(
          padding: const EdgeInsets.all(16),
          child: Stack(
            children: [
              RefreshIndicator(
                onRefresh: () async {
                  await context.read<ArtifactProvider>().refresh();
                  final list = context.read<ArtifactProvider>().artifacts;
                  final updated = list.firstWhere((a) => a.id == _artifact.id, orElse: () => _artifact);
                  setState(() => _artifact = updated);
                  await _loadHistory();
                },
                child: ListView(
                  children: [
                    _ReferenceImage(artifact: _artifact),
                    const SizedBox(height: 16),
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Expanded(
                          child: Text(
                            _artifact.name,
                            style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: AppColors.primary),
                          ),
                        ),
                        const SizedBox(width: 10),
                        StatusBadge(status: _artifact.status),
                      ],
                    ),
                    const SizedBox(height: 14),
                    // ── Thông tin cơ bản ──────────────────────────────────
                    Card(
                      child: Padding(
                        padding: const EdgeInsets.all(14),
                        child: Column(
                          children: [
                            _InfoRow(
                              icon: Icons.description_outlined,
                              title: 'Description',
                              value: (_artifact.description?.isNotEmpty == true) ? _artifact.description! : '—',
                            ),
                            const Divider(height: 22),
                            _InfoRow(
                              icon: Icons.place_outlined,
                              title: 'Location',
                              value: (_artifact.location?.isNotEmpty == true) ? _artifact.location! : '—',
                            ),
                            const Divider(height: 22),
                            _InfoRow(
                              icon: Icons.schedule_outlined,
                              title: 'Inspection Cycle',
                              value: _artifact.inspectionIntervalDays > 0
                                  ? 'Every ${_artifact.inspectionIntervalDays} days'
                                  : 'Not configured',
                            ),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: 12),
                    // ── Lần kiểm tra gần nhất ────────────────────────────
                    if (_historyLoading)
                      const Padding(
                        padding: EdgeInsets.symmetric(vertical: 12),
                        child: Center(child: CircularProgressIndicator(strokeWidth: 2)),
                      )
                    else if (_inspections.isNotEmpty)
                      _LastInspectionCard(inspection: _inspections.first),
                    const SizedBox(height: 14),
                    // ── Nút kiểm tra ──────────────────────────────────────
                    SizedBox(
                      width: double.infinity,
                      height: 52,
                      child: ElevatedButton.icon(
                        onPressed: _runInspection,
                        icon: const Icon(Icons.search),
                        label: const Text('Inspect via IoT Device'),
                      ),
                    ),
                    const SizedBox(height: 20),
                    // ── Lịch sử kiểm tra (inline) ────────────────────────
                    _buildHistorySection(),
                    const SizedBox(height: 24),
                  ],
                ),
              ),
              if (_busy)
                const Positioned.fill(
                  child: ColoredBox(
                    color: Color(0x66000000),
                    child: Center(child: CircularProgressIndicator(color: Colors.white)),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHistorySection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        InkWell(
          borderRadius: BorderRadius.circular(8),
          onTap: () => setState(() => _historyExpanded = !_historyExpanded),
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 6),
            child: Row(
              children: [
                const Icon(Icons.history, size: 18, color: AppColors.primary),
                const SizedBox(width: 8),
                Text(
                  _historyLoading
                      ? 'Inspection History'
                      : 'Inspection History (${_inspections.length})',
                  style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 15),
                ),
                const Spacer(),
                Icon(_historyExpanded ? Icons.expand_less : Icons.expand_more, color: AppColors.textMuted),
              ],
            ),
          ),
        ),
        if (_historyExpanded) ...[
          const SizedBox(height: 8),
          if (_historyLoading)
            const Center(child: Padding(padding: EdgeInsets.all(16), child: CircularProgressIndicator()))
          else if (_inspections.isEmpty)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(color: AppColors.surfaceMuted, borderRadius: BorderRadius.circular(12)),
              child: const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.history_toggle_off, color: AppColors.textFaint, size: 20),
                  SizedBox(width: 8),
                  Text('No inspections yet', style: TextStyle(color: AppColors.textMuted)),
                ],
              ),
            )
          else
            ...(_inspections.map((ins) => Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: _InlineHistoryItem(inspection: ins),
                ))),
        ],
      ],
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────

class _LastInspectionCard extends StatelessWidget {
  final Inspection inspection;
  const _LastInspectionCard({required this.inspection});

  @override
  Widget build(BuildContext context) {
    final score = inspection.damageScore;
    final Color scoreColor = score < 5
        ? AppColors.statusGood
        : score < 15
            ? AppColors.statusNeedCheck
            : AppColors.statusDamaged;
    return InkWell(
      borderRadius: BorderRadius.circular(16),
      onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => ResultScreen(inspection: inspection))),
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: scoreColor.withOpacity(0.3)),
          boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 8, offset: const Offset(0, 2))],
        ),
        child: Row(
          children: [
            Container(
              width: 44, height: 44,
              decoration: BoxDecoration(color: scoreColor.withOpacity(0.1), shape: BoxShape.circle),
              child: Icon(Icons.analytics_outlined, color: scoreColor, size: 22),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Latest Inspection', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 13)),
                  const SizedBox(height: 2),
                  Text(
                    DateFormat('HH:mm — dd/MM/yyyy').format(inspection.createdAt),
                    style: const TextStyle(color: AppColors.textMuted, fontSize: 12),
                  ),
                ],
              ),
            ),
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text('${score}%', style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: scoreColor)),
                StatusBadge(status: inspection.status, compact: true),
              ],
            ),
            const SizedBox(width: 4),
            const Icon(Icons.chevron_right, color: AppColors.textMuted, size: 18),
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────

class _InlineHistoryItem extends StatelessWidget {
  final Inspection inspection;
  const _InlineHistoryItem({required this.inspection});

  @override
  Widget build(BuildContext context) {
    final score = inspection.damageScore;
    final Color scoreColor = score < 5
        ? AppColors.statusGood
        : score < 15
            ? AppColors.statusNeedCheck
            : AppColors.statusDamaged;
    final thumbPath = inspection.annotatedImagePath ?? inspection.heatmapPath ?? inspection.currentImagePath;
    final thumbUrl = ApiConfig.resolveAssetUrl(thumbPath);
    return Card(
      margin: EdgeInsets.zero,
      child: InkWell(
        borderRadius: BorderRadius.circular(18),
        onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => ResultScreen(inspection: inspection))),
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              ClipRRect(
                borderRadius: BorderRadius.circular(10),
                child: thumbUrl.isNotEmpty
                    ? AuthenticatedImage(thumbUrl, width: 56, height: 56, fit: BoxFit.cover,
                        errorBuilder: (_, __, ___) => _fallbackThumb())
                    : _fallbackThumb(),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      DateFormat('HH:mm — dd/MM/yyyy').format(inspection.createdAt),
                      style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      inspection.inspectionType == InspectionType.scheduled ? 'Scheduled' : 'Ad-hoc',
                      style: TextStyle(
                        fontSize: 11,
                        color: inspection.inspectionType == InspectionType.scheduled ? Colors.blue : Colors.orange,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text('${score}%', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: scoreColor)),
                  const SizedBox(height: 4),
                  StatusBadge(status: inspection.status, compact: true),
                ],
              ),
              const SizedBox(width: 4),
              const Icon(Icons.chevron_right, color: AppColors.textMuted, size: 16),
            ],
          ),
        ),
      ),
    );
  }

  Widget _fallbackThumb() => Container(
        width: 56, height: 56,
        color: AppColors.surfaceMuted,
        child: const Icon(Icons.image_outlined, color: AppColors.textFaint),
      );
}

// ─────────────────────────────────────────────────────────────────────────────

class _ReferenceImage extends StatelessWidget {
  final Artifact artifact;
  const _ReferenceImage({required this.artifact});

  @override
  Widget build(BuildContext context) {
    final url = ApiConfig.resolveAssetUrl(artifact.referenceImagePath);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Reference Image (Golden Pose)', style: TextStyle(fontSize: 12, color: AppColors.textMuted)),
        const SizedBox(height: 4),
        AspectRatio(
          aspectRatio: 16 / 9,
          child: ClipRRect(
            borderRadius: BorderRadius.circular(20),
            child: artifact.hasImage && url.isNotEmpty
                ? AuthenticatedImage(url, fit: BoxFit.cover,
                    errorBuilder: (_, __, ___) => _placeholder(broken: true))
                : _placeholder(),
          ),
        ),
      ],
    );
  }

  Widget _placeholder({bool broken = false}) {
    return Container(
      color: AppColors.surfaceMuted,
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              broken ? Icons.broken_image_outlined : Icons.image_not_supported_outlined,
              size: 56,
              color: AppColors.textFaint,
            ),
            if (!broken)
              const Padding(
                padding: EdgeInsets.only(top: 8),
                child: Text(
                  'No reference image\n(Run golden pose initialization first)',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 12, color: AppColors.textFaint),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _InfoRow extends StatelessWidget {
  final IconData icon;
  final String title;
  final String value;
  const _InfoRow({required this.icon, required this.title, required this.value});

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, color: AppColors.primary, size: 20),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 2),
              Text(value, style: const TextStyle(color: AppColors.textMuted)),
            ],
          ),
        ),
      ],
    );
  }
}
