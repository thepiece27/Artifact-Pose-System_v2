-- Apply once to existing deployments. Inspections without a configured
-- baseline are stored as warning/unknown and must not self-reference the
-- current image as their previous image.
-- PostgreSQL:
ALTER TABLE image_comparisons
    ALTER COLUMN previous_image_id DROP NOT NULL;
