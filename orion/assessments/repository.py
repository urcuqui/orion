"""Assessment references use Orion's existing JSON/filesystem persistence."""
import json
import re
from pathlib import Path
from .model import Assessment


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', value):
        raise ValueError('invalid object identifier')
    return value


class AssessmentRepository:
    def __init__(self, base_dir='artifacts'):
        self.directory = Path(base_dir) / 'assessments'

    def get(self, assessment_id):
        path = self.directory / (identifier(assessment_id) + '.json')
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('assessment record must be an object')
        if data.get('assessment_id') != assessment_id:
            raise ValueError('assessment identifier does not match its record')
        return Assessment.from_dict(data)

    def save(self, assessment):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / (identifier(assessment.assessment_id) + '.json')
        # Replace one record atomically; no additional evidence is created.
        import uuid
        temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        try:
            temporary.write_text(json.dumps(assessment.to_dict(), indent=2), encoding='utf-8')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return assessment

    def list(self):
        records, errors = [], []
        if self.directory.exists():
            for path in self.directory.glob('*.json'):
                try:
                    records.append(self.get(path.stem))
                except (ValueError, TypeError, OSError) as exc:
                    errors.append({'assessment_id': path.stem, 'error': str(exc)})
        return sorted(records, key=lambda a: a.updated_at, reverse=True), errors
